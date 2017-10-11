//!

use errors::*;
use fold::{self, IdentityExistentialFolder, IdentityUniversalFolder, Fold, TypeFolder};
use ir::*;
use solve::infer::InferenceTable;

pub fn truncate<T>(infer: &mut InferenceTable,
                   universe: UniverseIndex,
                   max_size: usize,
                   value: &T)
                   -> Truncated<T::Result>
    where T: Fold,
{
    let mut truncater = Truncater::new(infer, universe, max_size);
    let value = value.fold_with(&mut truncater, 0).expect("Truncater is infallible");
    Truncated { overflow: truncater.overflow, value }
}

pub struct Truncated<T> {
    overflow: bool,
    value: T
}

struct Truncater<'infer> {
    infer: &'infer mut InferenceTable,
    universe: UniverseIndex,
    current_size: usize,
    max_size: usize,
    overflow: bool,
}

impl<'infer> Truncater<'infer> {
    fn new(infer: &'infer mut InferenceTable,
           universe: UniverseIndex,
           max_size: usize)
           -> Self
    {
        Truncater { infer, universe, current_size: 0, max_size, overflow: false }
    }

    fn overflow(&mut self, pre_size: usize) -> Ty {
        self.overflow = true;
        self.current_size = pre_size + 1;
        self.infer.new_variable(self.universe).to_ty()
    }
}

impl<'infer> TypeFolder for Truncater<'infer> {
    fn fold_ty(&mut self, ty: &Ty, binders: usize) -> Result<Ty> {
        let pre_size = self.current_size;
        self.current_size += 1;

        let result = fold::super_fold_ty(self, ty, binders)?;

        // We wish to maintain the invariant that:
        //
        //     pre_size < self.max_size =>
        //         post_size <= self.max_size
        //
        // Presuming that `pre_size < self.max_size`, then the
        // invariant is in jeopardy if `post_size > self.max_size`.
        // To repair the situation, we replace the entire subtree with
        // a fresh existential variable (in the innermost universe).
        let post_size = self.current_size;
        if pre_size < self.max_size && post_size > self.max_size {
            Ok(self.overflow(pre_size).up_shift(binders))
        } else {
            Ok(result)
        }
    }

    fn fold_lifetime(&mut self, lifetime: &Lifetime, binders: usize) -> Result<Lifetime> {
        fold::super_fold_lifetime(self, lifetime, binders)
    }
}

impl<'infer> IdentityExistentialFolder for Truncater<'infer> {
}

impl<'infer> IdentityUniversalFolder for Truncater<'infer> {
}

#[test]
fn truncate_types() {
    let mut table = InferenceTable::new();

    let environment0 = &Environment::new();
    let environment1 = &environment0.new_universe();
    let u1 = environment1.universe;
    let environment2 = &environment1.new_universe();

    // Vec<Vec<Vec<Vec<T>>>>
    let ty0 = ty!(apply (item 0)
                  (apply (item 0)
                   (apply (item 0)
                    (apply (item 0)
                     (apply (skol 1))))));

    // test: no truncation with size 5
    let Truncated { overflow, value: ty_no_overflow } = truncate(&mut table, u1, 5, &ty0);
    assert!(!overflow);
    assert_eq!(ty0, ty_no_overflow);

    // test: with size 3, truncates to `Vec<Vec<X>>`
    let ty_expect = ty!(apply (item 0)
                        (apply (item 0)
                         (var 0)));
    let Truncated { overflow, value: ty_overflow } = truncate(&mut table, u1, 3, &ty0);
    assert!(overflow);
    assert_eq!(ty_expect, ty_overflow);

    // test: the `X` is in u1, hence should fail to unify with a skolemized value in U2.
    let ty_in_u2 = ty!(apply (item 0)
                       (apply (item 0)
                        (apply (skol 2))));
    table.unify(environment2, &ty_overflow, &ty_in_u2).unwrap_err();
}
