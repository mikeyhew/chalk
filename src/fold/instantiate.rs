use ir::*;

use super::*;

pub struct Subst<'s> {
    /// Values to substitute. A reference to a free variable with
    /// index `i` will be mapped to `parameters[i]` -- if `i >
    /// parameters.len()`, then we will leave the variable untouched.
    parameters: &'s [Parameter]
}

impl<'s> Subst<'s> {
    pub fn new(parameters: &'s [Parameter]) -> Subst<'s> {
        Subst { parameters }
    }

    pub fn apply<T: Fold>(parameters: &[Parameter], value: &T) -> T::Result {
        value.fold_with(&mut Subst { parameters }, 0).unwrap()
    }
}

impl QuantifiedTy {
    pub fn subst(&self, parameters: &[Parameter]) -> Ty {
        assert_eq!(self.num_binders, parameters.len());
        self.ty.subst(parameters)
    }
}

impl<T: Fold> Binders<T> {
    pub fn instantiate_universally(&self, environment: &Arc<Environment>)
                                   -> InEnvironment<T::Result>
    {
        let mut new_environment = environment.clone();
        let parameters: Vec<_> =
            self.binders
                .iter()
                .map(|pk| {
                    new_environment = new_environment.new_universe();
                    match *pk {
                        ParameterKind::Lifetime(()) => {
                            let lt = Lifetime::ForAll(new_environment.universe);
                            ParameterKind::Lifetime(lt)
                        }
                        ParameterKind::Ty(()) =>
                            ParameterKind::Ty(Ty::Apply(ApplicationTy {
                                name: TypeName::ForAll(new_environment.universe),
                                parameters: vec![]
                            })),
                    }
                })
                .collect();
        let value = Subst::apply(&parameters, &self.value);
        InEnvironment::new(&new_environment, value)
    }
}

macro_rules! subst_method {
    ($t:ty) => {
        impl $t {
            pub fn subst(&self, parameters: &[Parameter]) -> Self {
                Subst::apply(parameters, self)
            }
        }
    }
}

subst_method!(Goal);
subst_method!(Ty);

impl<'b> ExistentialFolder for Subst<'b> {
    fn fold_free_existential_ty(&mut self, depth: usize, binders: usize) -> Result<Ty> {
        if depth >= self.parameters.len() {
            Ok(Ty::Var(depth - self.parameters.len() + binders))
        } else {
            match self.parameters[depth] {
                ParameterKind::Ty(ref t) => Ok(t.up_shift(binders)),
                _ => panic!("mismatched kinds in substitution"),
            }
        }
    }

    fn fold_free_existential_lifetime(&mut self, depth: usize, binders: usize) -> Result<Lifetime> {
        if depth >= self.parameters.len() {
            Ok(Lifetime::Var(depth - self.parameters.len() + binders))
        } else {
            match self.parameters[depth] {
                ParameterKind::Lifetime(ref l) => Ok(l.up_shift(binders)),
                _ => panic!("mismatched kinds in substitution"),
            }
        }
    }
}

impl<'b> IdentityUniversalFolder for Subst<'b> {
}
