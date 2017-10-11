//! Traits for transforming bits of IR.

use cast::Cast;
use errors::*;
use ir::*;
use std::fmt::Debug;
use std::sync::Arc;
use std::collections::BTreeMap;

mod instantiate;
mod shifted;
mod shifter;

pub use self::shifted::Shifted;
pub use self::shifter::Shifter;
pub use self::instantiate::Subst;

/// A "folder" is a transformer that can be used to make a copy of
/// some term -- that is, some bit of IR, such as a `Goal` -- with
/// certain changes applied. The idea is that it contains methods that
/// let you swap types/lifetimes for new types/lifetimes; meanwhile,
/// each bit of IR implements the `Fold` trait which, given a
/// `Folder`, will reconstruct itself, invoking the folder's methods
/// to transform each of the types/lifetimes embedded within.
///
/// # Usage patterns
///
/// ## Substituting for free variables
///
/// Most of the time, though, we are not interested in adjust
/// arbitrary types/lifetimes, but rather just free variables (even
/// more often, just free existential variables) that appear within
/// the term.
///
/// For this reason, the `Folder` trait extends two other traits that
/// contain methods that are invoked when just those particular
///
/// In particular, folders can intercept references to free variables
/// (either existentially or universally quantified) and replace them
/// with other types/lifetimes as appropriate.
///
/// To create a folder type F, one typically does one of two things:
///
/// - Implement `ExistentialFolder` and `IdentityUniversalFolder`:
///   - This ignores universally quantified variables but allows you to
///     replace existential variables with new values.
/// - Implement `ExistentialFolder` and `UniversalFolder`:
///   - This allows you to replace either existential or universal
///     variables with new types/lifetimes.
///
/// There is no reason to implement the `Folder` trait directly.
///
/// To **apply** a folder, use the `Fold::fold_with` method, like so
///
/// ```rust,ignore
/// let x = x.fold_with(&mut folder, 0);
/// ```
pub trait Folder: ExistentialFolder + UniversalFolder + TypeFolder {
    /// Returns a "dynamic" version of this trait. There is no
    /// **particular** reason to require this, except that I didn't
    /// feel like making `super_fold_ty` generic for no reason.
    fn to_dyn(&mut self) -> &mut Folder;
}

pub trait TypeFolder {
    fn fold_ty(&mut self, ty: &Ty, binders: usize) -> Result<Ty>;
    fn fold_lifetime(&mut self, lifetime: &Lifetime, binders: usize) -> Result<Lifetime>;
}

impl<T: ExistentialFolder + UniversalFolder + TypeFolder> Folder for T {
    fn to_dyn(&mut self) -> &mut Folder {
        self
    }
}

/// A convenience trait that indicates that this folder doesn't take
/// any action on types in particular, but just recursively folds
/// their contents (note that free variables that are encountered in
/// that process may still be substituted). The vast majority of
/// folders implement this trait.
pub trait DefaultTypeFolder {
}

impl<T: ExistentialFolder + UniversalFolder + DefaultTypeFolder> TypeFolder for T {
    fn fold_ty(&mut self, ty: &Ty, binders: usize) -> Result<Ty> {
        super_fold_ty(self.to_dyn(), ty, binders)
    }

    fn fold_lifetime(&mut self, lifetime: &Lifetime, binders: usize) -> Result<Lifetime> {
        super_fold_lifetime(self.to_dyn(), lifetime, binders)
    }
}

/// The methods for folding free **existentially quantified
/// variables**; for example, if you folded over `Foo<?T> }`, where `?T`
/// is an inference variable, then this would let you replace `?T` with
/// some other type.
pub trait ExistentialFolder {
    /// Invoked for `Ty::Var` instances that are not bound within the type being folded
    /// over:
    ///
    /// - `depth` is the depth of the `Ty::Var`; this is not adjusted to account for binders
    ///   in scope.
    /// - `binders` is the number of binders in scope.
    ///
    /// `depth >= binders` will always hold, since otherwise the type would be considered
    /// bound.
    ///
    /// This should return a type suitable for a context with `binders` in scope.
    fn fold_free_existential_ty(&mut self, depth: usize, binders: usize) -> Result<Ty>;

    /// As `fold_free_existential_ty`, but for lifetimes.
    fn fold_free_existential_lifetime(&mut self, depth: usize, binders: usize) -> Result<Lifetime>;
}

/// A convenience trait. If you implement this, you get an
/// implementation of `UniversalFolder` for free that simply ignores
/// universal values (that is, it replaces them with themselves).
pub trait IdentityExistentialFolder {
}

impl<T: IdentityExistentialFolder> ExistentialFolder for T {
    fn fold_free_existential_ty(&mut self, depth: usize, binders: usize) -> Result<Ty> {
        Ok(Ty::Var(depth + binders))
    }

    fn fold_free_existential_lifetime(&mut self, depth: usize, binders: usize) -> Result<Lifetime> {
        Ok(Lifetime::Var(depth + binders))
    }
}

pub trait UniversalFolder {
    /// Invoked for `Ty::Apply` instances where the type name is a `TypeName::ForAll`.
    /// Returns a type to use instead, which should be suitably shifted to account for `binders`.
    ///
    /// - `universe` is the universe of the `TypeName::ForAll` that was found
    /// - `binders` is the number of binders in scope
    fn fold_free_universal_ty(&mut self, universe: UniverseIndex, binders: usize) -> Result<Ty>;

    /// As with `fold_free_universal_ty`, but for lifetimes.
    fn fold_free_universal_lifetime(&mut self, universe: UniverseIndex, binders: usize) -> Result<Lifetime>;
}

/// A convenience trait. If you implement this, you get an
/// implementation of `UniversalFolder` for free that simply ignores
/// universal values (that is, it replaces them with themselves).
pub trait IdentityUniversalFolder {
}

impl<T: IdentityUniversalFolder> UniversalFolder for T {
    fn fold_free_universal_ty(&mut self, universe: UniverseIndex, _binders: usize) -> Result<Ty> {
        Ok(TypeName::ForAll(universe).to_ty())
    }

    fn fold_free_universal_lifetime(&mut self, universe: UniverseIndex, _binders: usize) -> Result<Lifetime> {
        Ok(universe.to_lifetime())
    }
}

/// Applies the given folder to a value.
pub trait Fold: Debug {
    /// The type of value that will be produced once folding is done.
    /// Typically this is `Self`, unless `Self` contains borrowed
    /// values, in which case owned values are produced (for example,
    /// one can fold over a `&T` value where `T: Fold`, in which case
    /// you get back a `T`, not a `&T`).
    type Result;

    /// Apply the given folder `folder` to `self`; `binders` is the
    /// number of binders that are in scope when beginning the
    /// folder. Typically `binders` starts as 0, but is adjusted when
    /// we encounter `Binders<T>` in the IR or other similar
    /// constructs.
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result>;
}

impl<'a, T: Fold> Fold for &'a T {
    type Result = T::Result;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        (**self).fold_with(folder, binders)
    }
}

impl<T: Fold> Fold for Vec<T> {
    type Result = Vec<T::Result>;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        self.iter().map(|e| e.fold_with(folder, binders)).collect()
    }
}

impl<T: Fold> Fold for Box<T> {
    type Result = Box<T::Result>;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        Ok(Box::new((**self).fold_with(folder, binders)?))
    }
}

impl<T: Fold> Fold for Arc<T> {
    type Result = Arc<T::Result>;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        Ok(Arc::new((**self).fold_with(folder, binders)?))
    }
}

macro_rules! tuple_fold {
    ($($n:ident),*) => {
        impl<$($n: Fold,)*> Fold for ($($n,)*) {
            type Result = ($($n::Result,)*);
            fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
                #[allow(non_snake_case)]
                let &($(ref $n),*) = self;
                Ok(($($n.fold_with(folder, binders)?,)*))
            }
        }
    }
}

tuple_fold!(A, B);
tuple_fold!(A, B, C);
tuple_fold!(A, B, C, D);

impl<T: Fold> Fold for Option<T> {
    type Result = Option<T::Result>;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        match *self {
            None => Ok(None),
            Some(ref e) => Ok(Some(e.fold_with(folder, binders)?)),
        }
    }
}

impl Fold for Ty {
    type Result = Self;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        folder.fold_ty(self, binders)
    }
}

pub fn super_fold_ty(folder: &mut Folder,
                     ty: &Ty,
                     binders: usize)
                     -> Result<Ty>
{
    match *ty {
        Ty::Var(depth) => if depth >= binders {
            folder.fold_free_existential_ty(depth - binders, binders)
        } else {
            Ok(Ty::Var(depth))
        },
        Ty::Apply(ref apply) => {
            let &ApplicationTy { name, ref parameters } = apply;
            match name {
                TypeName::ForAll(ui) => {
                    assert!(parameters.is_empty(),
                            "type {:?} with parameters {:?}",
                            ty,
                            parameters);
                    folder.fold_free_universal_ty(ui, binders)
                }

                TypeName::ItemId(_) | TypeName::AssociatedType(_) => {
                    let parameters = parameters.fold_with(folder, binders)?;
                    Ok(ApplicationTy { name, parameters }.cast())
                }
            }
        }
        Ty::Projection(ref proj) => Ok(Ty::Projection(proj.fold_with(folder, binders)?)),
        Ty::ForAll(ref quantified_ty) => Ok(Ty::ForAll(quantified_ty.fold_with(folder, binders)?)),
    }
}

impl Fold for QuantifiedTy {
    type Result = Self;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        let QuantifiedTy { num_binders, ref ty } = *self;
        Ok(QuantifiedTy { num_binders, ty: ty.fold_with(folder, binders + num_binders)? })
    }
}

impl<T> Fold for Binders<T>
    where T: Fold
{
    type Result = Binders<T::Result>;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        let Binders { binders: ref self_binders, value: ref self_value } = *self;
        let value = self_value.fold_with(folder, binders + self_binders.len())?;
        Ok(Binders { binders: self_binders.clone(), value: value })
    }
}

impl Fold for Lifetime {
    type Result = Self;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        folder.fold_lifetime(self, binders)
    }
}

pub fn super_fold_lifetime(folder: &mut Folder,
                           lifetime: &Lifetime,
                           binders: usize)
                           -> Result<Lifetime>
{
    match *lifetime {
        Lifetime::Var(depth) => if depth >= binders {
            folder.fold_free_existential_lifetime(depth - binders, binders)
        } else {
            Ok(Lifetime::Var(depth))
        },
        Lifetime::ForAll(universe) => {
            folder.fold_free_universal_lifetime(universe, binders)
        }
    }
}

impl Fold for Substitution {
    type Result = Substitution;
    fn fold_with(&self, folder: &mut Folder, binders: usize) -> Result<Self::Result> {
        let mut tys = BTreeMap::new();
        let mut lifetimes = BTreeMap::new();

        for (var, ty) in &self.tys {
            tys.insert(*var, ty.fold_with(folder, binders)?);
        }
        for (var, lt) in &self.lifetimes {
            lifetimes.insert(*var, lt.fold_with(folder, binders)?);
        }

        Ok(Substitution { tys, lifetimes })
    }
}

macro_rules! copy_fold {
    ($t:ty) => {
        impl ::fold::Fold for $t {
            type Result = Self;
            fn fold_with(&self,
                         _folder: &mut ::fold::Folder,
                         _binders: usize)
                         -> ::errors::Result<Self::Result> {
                Ok(*self)
            }
        }
    }
}

copy_fold!(Identifier);
copy_fold!(UniverseIndex);
copy_fold!(ItemId);
copy_fold!(usize);
copy_fold!(QuantifierKind);
// copy_fold!(TypeName); -- intentionally omitted! This is folded via `fold_ap`
copy_fold!(());

macro_rules! enum_fold {
    ($s:ident [$($n:ident),*] { $($variant:ident($($name:ident),*)),* } $($w:tt)*) => {
        impl<$($n),*> ::fold::Fold for $s<$($n),*> $($w)* {
            type Result = $s<$($n :: Result),*>;
            fn fold_with(&self,
                         folder: &mut ::fold::Folder,
                         binders: usize)
                         -> ::errors::Result<Self::Result> {
                match *self {
                    $(
                        $s::$variant( $(ref $name),* ) => {
                            Ok($s::$variant( $($name.fold_with(folder, binders)?),* ))
                        }
                    )*
                }
            }
        }
    }
}

enum_fold!(PolarizedTraitRef[] { Positive(a), Negative(a) });
enum_fold!(ParameterKind[T,L] { Ty(a), Lifetime(a) } where T: Fold, L: Fold);
enum_fold!(DomainGoal[] { Implemented(a), Normalize(a), WellFormed(a) });
enum_fold!(WellFormed[] { Ty(a), TraitRef(a) });
enum_fold!(LeafGoal[] { EqGoal(a), DomainGoal(a) });
enum_fold!(Constraint[] { LifetimeEq(a, b) });
enum_fold!(Goal[] { Quantified(qkind, subgoal), Implies(wc, subgoal), And(g1, g2), Not(g), Leaf(wc),
                    CannotProve(a) });

macro_rules! struct_fold {
    ($s:ident $([$($n:ident),*])* { $($name:ident),* } $($w:tt)*) => {
        impl $(<$($n),*>)* ::fold::Fold for $s $(<$($n),*>)* $($w)* {
            type Result = $s $(<$($n :: Result),*>)* ;
            fn fold_with(&self,
                         folder: &mut ::fold::Folder,
                         binders: usize)
                         -> ::errors::Result<Self::Result> {
                Ok($s {
                    $($name: self.$name.fold_with(folder, binders)?),*
                })
            }
        }
    }
}

struct_fold!(ProjectionTy { associated_ty_id, parameters });
struct_fold!(TraitRef { trait_id, parameters });
struct_fold!(Normalize { projection, ty });
struct_fold!(AssociatedTyValue { associated_ty_id, value });
struct_fold!(AssociatedTyValueBound { ty, where_clauses });
struct_fold!(Environment { universe, clauses });
struct_fold!(InEnvironment[F] { environment, goal } where F: Fold);
struct_fold!(EqGoal { a, b });
struct_fold!(ProgramClauseImplication { consequence, conditions });
struct_fold!(ConstrainedSubst { subst, constraints });
// struct_fold!(ApplicationTy { name, parameters }); -- intentionally omitted, folded through Ty
