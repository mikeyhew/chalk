//! An alternative solver based around the SLG algorithm, which
//! implements the well-formed semantics. This algorithm is very
//! closed based on the description found in the following paper,
//! which I will refer to in the comments as EWFS:
//!
//! > Efficient Top-Down Computation of Queries Under the Well-formed Semantics
//! > (Chen, Swift, and Warren; Journal of Logic Programming '95)
//!
//! However, to understand that paper, I would recommend first
//! starting with the following paper, which I will refer to in the
//! comments as NFTD:
//!
//! > A New Formulation of Tabled resolution With Delay
//! > (Swift; EPIA '99)
//!
//! Another useful paper that gives a kind of high-level overview of
//! concepts at play is the following, which I will refer to as XSB:
//!
//! > XSB: Extending Prolog with Tabled Logic Programming
//! > (Swift and Warren; Theory and Practice of Logic Programming '10)
//!
//! Glossary of other terms:
//!
//! - WAM: Warren abstract machine, an efficient way to evaluate Prolog programs.
//!   See <http://wambook.sourceforge.net/>.
//! - HH: Hereditary harrop predicates. What Chalk deals in.
//!   Popularized by Lambda Prolog.

use cast::{Cast, Caster};
use ir::*;
use ir::could_match::CouldMatch;
use solve::infer::{InferenceTable, UnificationResult};
use std::collections::{HashMap, HashSet};
use std::collections::hash_map::Entry;
use std::cmp::min;
use std::iter;
use std::mem;
use std::ops::{Index, IndexMut};
use std::sync::Arc;
use zip::Zip;

mod truncate;

pub fn solve_repl(overflow_depth: usize,
                  program: &Arc<ProgramEnvironment>,
                  goal: Goal)
                  -> Result<Answers, ExplorationError> {
    // To start, we peel off the `forall` and `exists` binders to create a canonical goal
    // with interesting questions.

    let mut infer = InferenceTable::new();
    let peeled_goal = {
        let mut env_goal = InEnvironment::new(&Environment::new(), goal);
        loop {
            let InEnvironment { environment, goal } = env_goal;
            match goal {
                Goal::Quantified(QuantifierKind::ForAll, subgoal) => {
                    let InEnvironment { environment, goal } =
                    subgoal.instantiate_universally(&environment);
                    env_goal = InEnvironment::new(&environment, *goal);
                }

                Goal::Quantified(QuantifierKind::Exists, subgoal) => {
                    let subgoal = infer.instantiate_in(environment.universe,
                                                       subgoal.binders.iter().cloned(),
                                                       &subgoal.value);
                    env_goal = InEnvironment::new(&environment, *subgoal);
                }

                _ => break InEnvironment::new(&environment, goal),
            }
        }
    };
    let canonical_goal = infer.canonicalize(&peeled_goal).quantified;
    Forest::solve_root_goal(overflow_depth, program, canonical_goal)
}

/// The **FOREST** of evaluation tracks all the in-progress work.
/// Conceptually, it corresponds to the forest F described in NFTD,
/// however, we structure it more like the "table" described in EWFS.
/// In particular, we never materialize the forest and subgraphs
/// *directly*, instead keeping two bits of information:
///
/// - There is **table** for each tree with root node `A :- A` in the forest.
///   This table is indexed by the (canonical) root node A. It contains
///   the answers found so far, as well as links to nodes from other trees in the
///   forest that are still waiting for answeres.
/// - There is a **stack** of nodes `A :- G` from the forest. Roughly
///   speaking, this stack stores nodes in the forest which have not
///   yet been completely evaluated.
///   - Calling this is stack can be a bit misleading: although the
///     activity of the system is focused on the top of the stack, we
///     also will wind up doing things like producing a new answer
///     that feeds into a goal higher-up the stack. For example, we might
///     have a stack like the following (where the stack grows down):
///
///         // foo(X) :- bar(X), baz(X).
///         // bar(X) :- ...
///
///     Here, we see that `foo(X)` is waiting on a result from `bar(X)`. Let's
///     say we just found an answer, `bar(1)`. In that case, we would feed that answer
///     to `foo`, causing us to push a new stack entry:
///
///         // foo(X) :- bar(X), baz(X).
///         // bar(X) :- ...
///         // foo(X) :- baz(1).
///
///     `bar(X)` and the node on top of it in the stack are not really
///     related. (Indeed, coping with this is actually the source of
///     some complexity in the machine itself.)
struct Forest {
    infer: InferenceTable,
    program: Arc<ProgramEnvironment>,
    dfn: DepthFirstNumber,
    tables: Tables,
    stack: Stack,
    overflow_depth: usize,
}

/// A unit type used to indicate that we have fully explored a
/// particular pathway.
struct FullyExplored;

/// The various kinds of errors we can encounter during exploration.
/// Note that these do not indicate **failed results** -- i.e, traits
/// not implemented. They also do not indicate the "third value" in
/// the WFS semantics. Rather they indicate that we could not figure
/// out the result for a given predicate in the WFS semantics (i.e.,
/// we could not prove, disprove, nor even find a definitive undefined
/// result).
#[derive(Debug)]
pub enum ExplorationError {
    /// Indicates that execution "flounded", meaning that it
    /// encountered a negative goal with unresolved variables.
    Floundered,
}

/// Indicates that we recursed too deeply.
#[derive(Copy, Clone, Debug)]
struct Overflow;

/// The result of exploration: either we fully explored some subtree,
/// populating the result tables with answers, or else we encountered
/// some kind of exploration error along the way.
type ExplorationResult = ::std::result::Result<FullyExplored, ExplorationError>;

/// See `Forest`.
#[derive(Default)]
struct Tables {
    /// Maps from a canonical goal to the index of its table.
    table_indices: HashMap<CanonicalGoal, TableIndex>,

    /// Table: as described above, stores the key information for each
    /// tree in the forest.
    tables: Vec<Table>,
}

struct TruthValues<'a> {
    /// The tables, which all must be completely evaluated.
    tables: &'a Tables,

    /// Truth values for each table; initially populated with `None`.
    truth_values: Vec<Option<TruthValue>>,
}

/// See `Forest`.
#[derive(Default)]
struct Stack {
    /// Stack: as described above, stores the in-progress goals.
    stack: Vec<StackEntry>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct TableIndex {
    value: usize
}

copy_fold!(TableIndex);

/// The StackIndex identifies the position of a table's goal in the
/// stack of goals that are actively being processed. Note that once a
/// table is completely evaluated, it may be popped from the stack,
/// and hence no longer have a stack index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct StackIndex {
    value: usize
}

copy_fold!(StackIndex);

/// The `DepthFirstNumber` (DFN) is a sequential number assigned to
/// each goal when it is first encountered. The naming (taken from
/// EWFS) refers to the idea that this number tracks the index of when
/// we encounter the goal during a depth-first traversal of the proof
/// tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct DepthFirstNumber {
    value: u64
}

copy_fold!(DepthFirstNumber);

struct StackEntry {
    /// The goal G from the stack entry `A :- G` represented here.
    table: TableIndex,

    /// The DFN of this computation.
    dfn: DepthFirstNumber,

    /// Tracks the dependencies of this stack entry on things beneath
    /// it in the stack. This field is updated "periodically",
    /// e.g. when a direct subgoal completes. Otherwise, the minimums
    /// for the active computation are tracked in a local variable
    /// that is threaded around.
    ///
    /// Note that this field is an over-approximation. As described in
    /// section 3.4.1 of EWFS, it actually stores the minimal
    /// dependencies of this stack entry **and anything on top of it
    /// in the stack**. In some cases, it can happen that this entry
    /// on the stack does not depend on the things on top of it, in
    /// which case the `link` is overapproximated -- this
    /// overapproximation reflects the fact that, because of the
    /// nature of a stack, we cannot in fact pop this entry until
    /// those other entries are popped, even though there is no
    /// *logical* dependency between us. This is the price we pay for
    /// using such a simple data structure.
    link: Minimums,
}

struct Table {
    /// Stores the answers that we have found thus far. For each head
    /// goal, we store a set of "delayed literals" instances. So, if
    /// the SLG algorithm would have computed (e.g.) `A :- ~B |` and
    /// `A :- ~C |` as answers, this would be stored as a `A -> {{B},
    /// {C}}` map entry.
    answers: HashMap<CanonicalGoal, DelayedLiteralsSet>,

    /// These are answers that have not been fully proven because we
    /// encountered overflow trying to prove them. Obviously they may
    /// or may not be true. =)
    overflow_answers: HashSet<CanonicalGoal>,

    /// Stack entries waiting to hear about POSITIVE results from this
    /// table. This occurs when you have something like `foo(X) :-
    /// bar(X)`.
    positives: Vec<CanonicalPendingExClause>,

    /// Stack entries waiting to hear about NEGATIVE results from this
    /// table. This occurs when you have something like `foo(X) :- not
    /// bar(X)`.
    negatives: Vec<CanonicalPendingExClause>,

    /// Stores the index of this table on the stack. This is only
    /// `Some` until the table has been COMPLETELY EVALUATED -- i.e.,
    /// all possible answers have been found -- at which point it is
    /// set to `None`.
    depth: Option<StackIndex>,
}

/// A truth value in the WFS.
#[derive(Copy, Clone, Debug)]
enum TruthValue {
    /// Contains a finite proof.
    True,

    /// Contains no proof or an infinite proof.
    False,

    /// Participates in a negative cycle.
    ///
    /// Consider this: are `a` and `b` true if `a :- not b. b :- not a.`?
    Unknown,
}

#[derive(Clone, Debug)]
struct PendingExClause {
    goal_depth: StackIndex,
    table_goal: InEnvironment<Goal>,
    selected_goal: InEnvironment<Goal>,
    delayed_literals: Vec<InEnvironment<Goal>>,
    subgoals: Vec<Literal>,
}

struct_fold!(PendingExClause {
    goal_depth,
    table_goal,
    selected_goal,
    delayed_literals,
    subgoals
});

/// The paper describes these as `A :- D | G`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExClause {
    /// The goal of the table `A`.
    table_goal: InEnvironment<Goal>,

    /// Delayed literals: things that we depend on negatively,
    /// but which have not yet been fully evaluated.
    delayed_literals: Vec<InEnvironment<Goal>>,

    /// Subgoals: literals that must be proven
    subgoals: Vec<Literal>,
}

struct_fold!(ExClause { table_goal, delayed_literals, subgoals });

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Answers {
    pub answers: Vec<Answer>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Answer {
    goal: CanonicalGoal,
    overflow: bool,
    delayed_literals: Vec<CanonicalGoal>,
}

#[derive(Clone, Debug)]
enum DelayedLiteralsSet {
    None,
    Some(HashSet<DelayedLiterals>),
}

/// A set of delayed literals. The vector in this struct must
/// meet two conditions:
///
/// 1. It is sorted, ensuring that we don't have to worry about permutations.
/// 2. The goals within cannot contain any free variables; in other
///    words, no inference variables.
///
/// (We still store canonical versions of the literals, so as to
/// guarantee that all inference variables have been fully resolved,
/// but the set of binders will always be empty.)
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DelayedLiterals {
    delayed_literals: Vec<CanonicalGoal>
}

/// Either `A` or `~A`, where `A` is a `DomainGoal`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Literal {
    Positive(InEnvironment<Goal>),
    Negative(InEnvironment<Goal>),
}

enum_fold!(Literal[] { Positive(a), Negative(a) });

/// A set of logical conditions, all of which must be true.
struct Product {
    literals: Vec<Literal>
}

/// The `Minimums` structure is used to track the dependencies between
/// some item E on the evaluation stack. In particular, it tracks
/// cases where the success of E depends (or may depend) on items
/// deeper in the stack than E (i.e., with lower DFNs).
///
/// `positive` tracks the lowest index on the stack to which we had a
/// POSITIVE dependency (e.g. `foo(X) :- bar(X)`) -- meaning that in
/// order for E to succeed, the dependency must succeed. It is
/// initialized with the index of the predicate on the stack. So
/// imagine we have a stack like this:
///
///     // 0 foo(X)   <-- bottom of stack
///     // 1 bar(X)
///     // 2 baz(X)   <-- top of stack
///
/// In this case, `positive` would be initially 0, 1, and 2 for `foo`,
/// `bar`, and `baz` respectively. This reflects the fact that the
/// answers for `foo(X)` depend on the answers for `foo(X)`. =)
///
/// Now imagine that we had a clause `baz(X) :- foo(X)`, inducing a
/// cycle. In this case, we would update `positive` for `baz(X)` to be
/// 0, reflecting the fact that its answers depend on the answers for
/// `foo(X)`. Similarly, the minimum for `bar` would (eventually) be
/// updated, since it too transitively depends on `foo`. `foo` is
/// unaffected.
///
/// `negative` tracks the lowest index on the stack to which we had a
/// NEGATIVE dependency (e.g., `foo(X) :- not { bar(X) }`) -- meaning
/// that for E to succeed, the dependency must fail. This is initially
/// `usize::MAX`, reflecting the fact that the answers for `foo(X)` do
/// not depend on `not(foo(X))`. When negative cycles are encountered,
/// however, this value must be updated.
#[derive(Copy, Clone, Debug)]
struct Minimums {
    positive: DepthFirstNumber,
    negative: DepthFirstNumber,
}

#[derive(Copy, Clone, Debug)]
enum Sign {
    Positive,
    Negative,
}

#[derive(Copy, Clone, Debug)]
enum Satisfiable<T> {
    Yes(T),
    No
}

type CanonicalGoal = Canonical<InEnvironment<Goal>>;
type CanonicalDomainGoal = Canonical<InEnvironment<DomainGoal>>;
type CanonicalSubst = Canonical<ConstrainedSubst>;
type CanonicalExClause = Canonical<ExClause>;
type CanonicalPendingExClause = Canonical<PendingExClause>;

impl Forest {
    fn solve_root_goal(overflow_depth: usize,
                       program: &Arc<ProgramEnvironment>,
                       root_goal: CanonicalGoal)
                       -> Result<Answers, ExplorationError> {
        let program = program.clone();

        let mut forest = Forest {
            overflow_depth: overflow_depth,
            infer: InferenceTable::new(),
            dfn: DepthFirstNumber::MIN,
            program: program.clone(),
            tables: Tables::default(),
            stack: Stack::default(),
        };

        let (root_table, root_table_depth) = forest.push_new_table(&root_goal, None, None);
        let mut minimums = forest.stack[root_table_depth].link;
        let instantiated_goal = forest.infer.instantiate_canonical(&root_goal);
        forest.subgoal(root_table_depth, instantiated_goal, &mut minimums)?;

        let mut truth_values = TruthValues::new(&forest.tables);
        let answers = truth_values.simplified_answers(root_table);
        Ok(Answers { answers })
    }

    /// Pushes a new goal onto the stack, creating a table entry in the process.
    fn push_new_table(&mut self,
                      goal: &CanonicalGoal,
                      positive_pending: Option<CanonicalPendingExClause>,
                      negative_pending: Option<CanonicalPendingExClause>)
                      -> (TableIndex, StackIndex) {
        let depth = self.stack.next_index();
        let dfn = self.dfn.next();
        let table = self.tables.insert(goal, depth);
        debug!("push_new_table: depth {:?} is table {:?} with goal {:?}", depth, table, goal);
        self.tables[table].positives.extend(positive_pending);
        self.tables[table].negatives.extend(negative_pending);
        self.stack.push(table, dfn);
        (table, depth)
    }

    /// Creates an inference snapshot and executes `op`, rolling back
    /// the snapshot afterwards. This is generally safe to use in any
    /// context where we are doing exploring (hence the return type),
    /// since -- due to the nature of the EWFS algorithm -- any result
    /// that may escape a stack frame (e.g., by being stored in a
    /// table) is canonicalized first.
    fn snapshotted<F>(&mut self, op: F) -> ExplorationResult
        where F: FnOnce(&mut Self) -> ExplorationResult
    {
        let snapshot = self.infer.snapshot();
        let result = op(self);
        self.infer.rollback_to(snapshot);
        result
    }

    /// This is SLG_SUBGOAL from EWFS. It is invoked when a new goal
    /// has been freshly pushed. We do a slight tweak to account for
    /// HH vs domain goals.
    fn subgoal(&mut self,
               goal_depth: StackIndex,
               goal: InEnvironment<Goal>,
               minimums: &mut Minimums)
               -> ExplorationResult
    {
        debug_heading!("subgoal(goal_depth={:?}, goal={:?}, minimums={:?})",
                       goal_depth, goal, minimums);

        // We want to consider two cases:
        //
        // - The goal is a domain goal. In that case, we will make N alternatives,
        //   one for each clause we can find.
        // - The goal is some other kind of HH goal. In that case, we will break it
        //   down into a product of literals, and create 1 alternative.

        let InEnvironment { environment, goal } = goal;
        match goal {
            Goal::Leaf(LeafGoal::DomainGoal(domain_goal)) => {
                let domain_goal = InEnvironment::new(&environment, domain_goal);
                let clauses = self.clauses(&domain_goal);

                for clause in clauses {
                    self.snapshotted(|this| {
                        match Self::resolvent_clause(&mut this.infer,
                                                     &domain_goal,
                                                     &clause.implication) {
                            Satisfiable::No =>
                                Ok(FullyExplored),
                            Satisfiable::Yes(resolvent) =>
                                this.new_clause(goal_depth, resolvent, minimums),
                        }
                    })?;
                }
            }

            _ => {
                // `canonical_goal` is an HH goal. We can simplify it
                // into a series of *literals*, all of which must be
                // true.
                let hh_goal = InEnvironment::new(&environment, goal);
                let product = match Self::simplify_hh_goal(&mut self.infer, &hh_goal) {
                    Satisfiable::Yes(product) => product,
                    Satisfiable::No => return Ok(FullyExplored), // now way to solve
                };

                // Construct an ex-clause. In EWFS, this is marked as
                //
                //     A :- A
                let ex_clause = ExClause {
                    table_goal: hh_goal,
                    delayed_literals: vec![],
                    subgoals: product.literals,
                };

                self.new_clause(goal_depth, ex_clause, minimums)?;
            }
        }

        debug!("subgoal: goal_depth={:?} minimums={:?}", goal_depth, minimums);
        self.complete(goal_depth, minimums)
    }

    /// Returns all clauses that are relevant to `goal`, either from
    /// the environment or the program.
    fn clauses(&mut self,
               goal: &InEnvironment<DomainGoal>)
               -> Vec<ProgramClause>
    {
        let &InEnvironment { ref environment, ref goal } = goal;

        let environment_clauses =
            environment.clauses
                       .iter()
                       .filter(|&env_clause| env_clause.could_match(goal))
                       .map(|env_clause| env_clause.clone().into_program_clause());

        let program_clauses =
            self.program.program_clauses.iter()
                                        .filter(|clause| clause.could_match(goal))
                                        .cloned();

        environment_clauses.chain(program_clauses).collect()
    }

    /// Pop off the next subgoal from `ex_clause` and try to solve
    /// it. Invoked when we have either just started a fresh goal (and
    /// selected a program clause) or when we have a new answer to a
    /// blocked goal that has just been incorporated.
    fn new_clause(&mut self,
                  goal_depth: StackIndex,
                  mut ex_clause: ExClause, // Contains both A and G together.
                  minimums: &mut Minimums)
                  -> ExplorationResult
    {
        debug_heading!("new_clause(goal_depth={:?}, ex_clause={:?}, minimums={:?}",
                       goal_depth, ex_clause, minimums);

        self.snapshotted(|this| {
            if this.stack.len() > this.overflow_depth {
                debug!("overflow");
                return this.overflow_answer(goal_depth, &ex_clause.table_goal);
            }

            match ex_clause.subgoals.pop() {
                // No goals left to prove: this is an answer.
                None =>
                    this.answer(goal_depth, ex_clause, minimums),

                // Positive goal.
                Some(Literal::Positive(selected_goal)) =>
                    this.positive(goal_depth, ex_clause, selected_goal, minimums),

                // Negative goal. EWFS checks for whether `selected_goal`
                // is ground here. We push this check into `negative`.
                Some(Literal::Negative(selected_goal)) =>
                    this.negative(goal_depth, ex_clause, selected_goal, minimums),
            }
        })
    }

    /// Try to solve a positive selected literal.
    fn positive(&mut self,
                goal_depth: StackIndex,
                ex_clause: ExClause,
                selected_goal: InEnvironment<Goal>,
                minimums: &mut Minimums)
                -> ExplorationResult
    {
        debug_heading!(
            "positive(goal_depth={:?}, ex_clause={:?}, selected_goal={:?}, minimums={:?})",
            goal_depth, ex_clause, selected_goal, minimums);

        let canonical_literal: CanonicalGoal =
            self.infer.canonicalize(&selected_goal).quantified;

        debug!("positive: canonical_literal={:?}", canonical_literal);

        // Check if we have an existing table for this selected literal.
        let subgoal_table = match self.tables.index_of(&canonical_literal) {
            Some(i) => i,
            None => {
                // If not, that's the easy case. Start a new table, listing ourselves
                // as being pending.
                let pending_ex_clause = self.pending_ex_clause(goal_depth,
                                                               &ex_clause,
                                                               &selected_goal);
                let (subgoal_table, subgoal_depth) = self.push_new_table(&canonical_literal,
                                                                         Some(pending_ex_clause),
                                                                         None);
                let mut subgoal_minimums = self.stack.top().link;
                self.subgoal(subgoal_depth, selected_goal, &mut subgoal_minimums)?;
                self.update_solution(goal_depth,
                                     subgoal_table,
                                     Sign::Positive,
                                     minimums,
                                     &mut subgoal_minimums);
                return Ok(FullyExplored);
            }
        };

        // A table for this entry already exists. We want to take
        // whatever answers we can find in the table -- bearing in
        // mind that the table may still be in the process of being
        // evaluated!
        if let Some(subgoal_depth) = self.tables[subgoal_table].depth {
            // If the table is not completely evaluated, then there is
            // a cycle.  We'll still use whatever answers have been
            // found so far, but we'll also register ourselves to
            // receive any new answers that will come later.
            let pending_ex_clause = self.pending_ex_clause(goal_depth,
                                                           &ex_clause,
                                                           &selected_goal);
            self.tables[subgoal_table].positives.push(pending_ex_clause);
            self.update_lookup(goal_depth, subgoal_depth, Sign::Positive, minimums);
        }

        // First, process the overflow answers.
        let new_overflows: Vec<_> = {
            let infer = &mut self.infer;
            self.tables[subgoal_table]
                .overflow_answers
                .iter()
                .filter_map(|answer| Self::overflow(infer, &ex_clause, &selected_goal, answer).yes())
                .collect()
        };

        for new_overflow_answer in new_overflows {
            self.overflow_answer(goal_depth, &new_overflow_answer)?;
        }

        // Next, we will process the answers that have already been
        // found one by one.
        let new_ex_clauses: Vec<_> = {
            let infer = &mut self.infer;
            self.tables[subgoal_table]
                .answers
                .iter()
                .filter_map(|(answer_goal, answer_delayed_literals)| {
                    Self::incorporate_cached_answer(infer,
                                                    &ex_clause,
                                                    &selected_goal,
                                                    answer_goal,
                                                    answer_delayed_literals)
                        .yes()
                })
                .collect()
        };

        for ex_clause in new_ex_clauses {
            let snapshot = self.infer.snapshot();
            self.new_clause(goal_depth, ex_clause, minimums)?;
            self.infer.rollback_to(snapshot);
        }

        Ok(FullyExplored)
    }

    /// Creates a `PendingExClause` representing the current node in the forest.
    ///
    /// # Parameters
    ///
    /// - `goal_depth` -- the depth of the suspended goal in the stack
    /// - `ex_clause` -- the thing we are trying to prove (`A |- G` in EWFS),
    ///   but with selected literal popped
    /// - `selected_goal` -- the selected literal. This could be either positive
    ///   or negative depending on context.
    fn pending_ex_clause(&mut self,
                         goal_depth: StackIndex,
                         ex_clause: &ExClause,
                         selected_goal: &InEnvironment<Goal>)
                         -> CanonicalPendingExClause
    {
        let parts = (
            &ex_clause.table_goal,
            &selected_goal,
            &ex_clause.delayed_literals,
            &ex_clause.subgoals
        );
        let canonical_parts = self.infer.canonicalize(&parts).quantified;
        canonical_parts.map(|(table_goal, selected_goal, delayed_literals, subgoals)| {
            PendingExClause {
                goal_depth, table_goal, selected_goal, delayed_literals, subgoals
            }
        })
    }

    fn negative(&mut self,
                goal_depth: StackIndex,
                mut ex_clause: ExClause,
                selected_goal: InEnvironment<Goal>,
                minimums: &mut Minimums)
                -> ExplorationResult
    {
        debug_heading!(
            "negative(goal_depth={:?}, ex_clause={:?}, selected_goal={:?}, minimums={:?})",
            goal_depth, ex_clause, selected_goal, minimums);

        // We want to prove ~B where B == `selected_goal`.
        let canonical_literal = match self.infer.negated(&selected_goal) {
            Some(g) => g,
            None => {
                // Not ground. For now, abort, though conceivably it would
                // make sense to "defer" this goal yet further until other
                // subgoals have been processed. Probably the best way to
                // do that would to split the pos/neg sub-goals into
                // distinct vectors, but I'm not smart enough to do that
                // just now.
                return Err(ExplorationError::Floundered);
            }
        };

        // Check whether B has a table already.
        let subgoal_table = match self.tables.index_of(&canonical_literal) {
            Some(i) => i,
            None => {
                // No table. Create one and start solving B.
                let pending_ex_clause = self.pending_ex_clause(goal_depth,
                                                               &ex_clause,
                                                               &selected_goal);
                let (subgoal_table, subgoal_depth) = self.push_new_table(&canonical_literal,
                                                                         None,
                                                                         Some(pending_ex_clause));
                let mut subgoal_minimums = self.stack.top().link;
                self.subgoal(subgoal_depth, selected_goal, &mut subgoal_minimums)?;
                self.update_solution(goal_depth,
                                     subgoal_table,
                                     Sign::Negative,
                                     minimums,
                                     &mut subgoal_minimums);
                return Ok(FullyExplored);
            }
        };

        // If we already know that the subgoal is satisfiable, we can
        // stop now.
        if self.tables[subgoal_table].is_satisfiable() {
            return Ok(FullyExplored);
        }

        if let Some(subgoal_depth) = self.tables[subgoal_table].depth {
            // Not yet completely evaluated. Register ourselves as
            // having interest in negative solutions and stop for now.
            let pending_ex_clause = self.pending_ex_clause(goal_depth,
                                                           &ex_clause,
                                                           &selected_goal);
            self.tables[subgoal_table].negatives.push(pending_ex_clause);
            self.update_lookup(goal_depth, subgoal_depth, Sign::Negative, minimums);
            return Ok(FullyExplored);
        }

        // The table for B is completely evaluated and not known to be
        // satisfiable; but there be answers with delayed literals
        // still. If so, we have to delay B ourselves (otherwise, ~B
        // is proven).
        if {
            !self.tables[subgoal_table].overflow_answers.is_empty() ||
                !self.tables[subgoal_table].answers.is_empty()
        } {
            ex_clause.delayed_literals.push(selected_goal);
        }

        // Proceed to the remaining subgoals.
        self.new_clause(goal_depth, ex_clause, minimums)
    }

    fn incorporate_cached_answer(infer: &mut InferenceTable,
                                 ex_clause: &ExClause,
                                 selected_goal: &InEnvironment<Goal>,
                                 answer_goal: &CanonicalGoal,
                                 answer_delayed_literals: &DelayedLiteralsSet)
                                 -> Satisfiable<ExClause>
    {
        match *answer_delayed_literals {
            DelayedLiteralsSet::None =>
                Self::resolvent_answer(infer, ex_clause, selected_goal, answer_goal),
            DelayedLiteralsSet::Some(_) =>
                Self::factor(infer, ex_clause, selected_goal, answer_goal),
        }
    }

    fn answer(&mut self,
              goal_depth: StackIndex,
              ex_clause: ExClause, // Contains both A and G together.
              minimums: &mut Minimums)
              -> ExplorationResult
    {
        debug_heading!(
            "answer(goal_depth={:?}, ex_clause={:?}, minimums={:?})",
            goal_depth, ex_clause, minimums);

        let goal_table = self.stack[goal_depth].table;

        // Decompose the answer.
        let ExClause { table_goal, delayed_literals, subgoals } = ex_clause;
        assert!(subgoals.is_empty());

        // Count how many answers this table has so far. If it exceeds
        // our overflow threshold, then convert this answer into an
        // OVERFLOW ANSWER. In the case where there are no delayed
        // literals, this is a bit inaccurate: we *know* that this
        // answer holds, after all. But converting to an overflow goal
        // prevents us from propagating across positive links, cutting
        // off feedback loops that produce more and more answers.
        if self.tables[goal_table].answers.len() > self.overflow_depth {
            return self.overflow_answer(goal_depth, &table_goal);
        }

        // Produce the canonical form of the answer.
        let answer_goal = self.infer.canonicalize(&table_goal).quantified;
        debug!("answer: goal_table={:?}, answer_goal={:?}", goal_table, answer_goal);

        // Convert the `DelayedLiterals` instance representing the set
        // of delayed literals from this ex-clause.
        let delayed_literals = {
            let mut delayed_literals: Vec<_> =
                delayed_literals.into_iter()
                                .map(|g| self.infer.canonicalize(&g).quantified)
                                .inspect(|g| assert!(g.binders.is_empty()))
                                .collect();
            delayed_literals.sort();
            DelayedLiterals { delayed_literals }
        };
        debug!("answer: delayed_literals={:?}", delayed_literals);

        // (*) NB: delayed literals cannot have free inference variables

        // Determine if answer is new. If so, insert and notify.
        let list: Vec<_> = if delayed_literals.delayed_literals.is_empty() {
            debug!("answer: no delayed_literals, previous answer = {:?}",
                   self.tables[goal_table].answers.get(&answer_goal));

            // If we already saw an answer with no delayed literals,
            // stop. Otherwise, continue.
            match self.tables[goal_table].answers.entry(answer_goal.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(DelayedLiteralsSet::None);
                }

                Entry::Occupied(mut entry) => {
                    if let &DelayedLiteralsSet::None = entry.get() {
                        return Ok(FullyExplored);
                    }

                    entry.insert(DelayedLiteralsSet::None);
                }
            }

            // Clear out all the people waiting for negative results; we
            // have an answer now, so they have failed.
            self.tables[goal_table].negatives = vec![];

            // Produce a list of people waiting for *positive* results.
            let infer = &mut self.infer;
            self.tables[goal_table]
                .positives
                .iter()
                .filter_map(|p| Self::resolvent_pending(infer, p, &answer_goal).yes())
                .collect()
        } else {
            debug!("answer: has delayed_literals, previous answer = {:?}",
                   self.tables[goal_table].answers.get(&answer_goal));

            if let Some(entry) = self.tables[goal_table].answers.get_mut(&answer_goal) {
                // Already have an entry with this head atom. No
                // need to notify pending people, since they don't
                // care about the details of what the delayed
                // literals are.

                match *entry {
                    DelayedLiteralsSet::None => {
                        // We don't care about answers with
                        // delayed literals if we already had an
                        // unconditional answer.
                    }

                    DelayedLiteralsSet::Some(ref mut v) => {
                        // We don't care about answers with
                        // delayed literals if we already had an
                        // unconditional answer.
                        v.insert(delayed_literals);
                    }
                }

                return Ok(FullyExplored);
            }

            // No entry yet with this head atom. We need to
            // notify pending people, so don't return.
            self.tables[goal_table].answers.insert(
                answer_goal.clone(),
                DelayedLiteralsSet::Some(iter::once(delayed_literals).collect()));

            let infer = &mut self.infer;
            self.tables[goal_table]
                .positives
                .iter()
                .filter_map(|p| Self::factor_pending(infer, p, &answer_goal).yes())
                .collect()
        };

        // Process each of them in turn.
        for (pending_table, pending_ex_clause) in list {
            self.new_clause(pending_table, pending_ex_clause, minimums)?;
        }

        Ok(FullyExplored)
    }

    fn overflow_answer(&mut self,
                       goal_depth: StackIndex,
                       overflow_goal: &InEnvironment<Goal>)
                       -> ExplorationResult
    {
        debug_heading!(
            "overflow_answer(goal_depth={:?}, overflow_goal{:?})",
            goal_depth, overflow_goal);

        let answer_goal = self.infer.canonicalize(&overflow_goal).quantified;
        let goal_table = self.stack[goal_depth].table;

        debug!("overflow_answer: answer_goal = {:?}", answer_goal);

        if !self.tables[goal_table].overflow_answers.insert(answer_goal.clone()) {
            // we've already seen this overflow answer
            return Ok(FullyExplored);
        }

        // Push the overflow answer to our positive dependents,
        // creating follow-on overflow answers.

        // Clear out all the people waiting for negative results; we
        // have an overflow answer now, so they will never be fully
        // resolved (we will wind up as a delayed literal for them).
        self.tables[goal_table].negatives = vec![];

        let new_overflow_answers: Vec<_> = {
            let infer = &mut self.infer;
            self.tables[goal_table]
                .positives
                .iter()
                .filter_map(|p| {
                    debug!("overflow_answer: positive = {:?}", answer_goal);
                    Self::overflow_pending(infer, p, &answer_goal).yes()
                })
                .collect()
        };

        for (new_table, new_overflow_answer) in new_overflow_answers {
            self.overflow_answer(new_table, &new_overflow_answer)?;
        }

        Ok(FullyExplored)
    }

    /// Updates `minimums` to account for the dependencies of a
    /// subgoal. Invoked when:
    ///
    /// - in the midst of solving `table`,
    /// - `subgoal_table` was the selected literal,
    /// - we invoked `subgoal()` and it returned,
    /// - with `subgoal_minimums` as its "result".
    fn update_solution(&mut self,
                       goal_depth: StackIndex,
                       subgoal_table: TableIndex,
                       sign: Sign,
                       minimums: &mut Minimums,
                       subgoal_minimums: &Minimums)
    {
        debug!(
            "update_solution(goal_depth={:?}, subgoal_table={:?}, sign={:?}, \
             minimums={:?}, subgoal_minimums={:?})",
            goal_depth, subgoal_table, sign, minimums, subgoal_minimums);

        if let Some(subgoal_depth) = self.tables[subgoal_table].depth {
            self.update_lookup(goal_depth, subgoal_depth, sign, minimums);
        } else {
            self.stack[goal_depth].link.take_minimums(subgoal_minimums);
            minimums.take_minimums(subgoal_minimums);
        }
    }

    /// Like `update_solution`, but invoked when `subgoal_table`
    /// is known to be incomplete.
    fn update_lookup(&mut self,
                     goal_depth: StackIndex,
                     subgoal_depth: StackIndex,
                     sign: Sign,
                     minimums: &mut Minimums)
    {
        match sign {
            Sign::Positive => {
                let subgoal_link = self.stack[subgoal_depth].link;
                self.stack[goal_depth].link.take_minimums(&subgoal_link);
                minimums.take_minimums(&subgoal_link);
            }

            Sign::Negative => {
                // If `goal` depends on `not(subgoal)`, then for goal
                // to succeed, `subgoal` must be completely
                // evaluated. Therefore, `goal` depends (negatively)
                // on the minimum link of `subgoal` as a whole -- it
                // doesn't matter whether it's pos or neg.
                let subgoal_min = self.stack[subgoal_depth].link.minimum_of_pos_and_neg();
                self.stack[goal_depth].link.take_negative_minimum(subgoal_min);
                minimums.take_negative_minimum(subgoal_min);
            }
        }
    }

    /// This method is invoked each time we exhaust all of the
    /// possibilities for exploration at the point in the stack
    /// (`goal_depth`). This doesn't mean that we are finished
    /// with the goal: for example, there may be a cycle, like
    ///
    /// ```notrust
    /// s :- p
    /// s :- ...
    /// p :- s // <-- when `complete` is invoked on `q`, we will not be done
    ///               exploring `s`
    /// ```
    fn complete(&mut self,
                completed_goal_depth: StackIndex,
                minimums: &mut Minimums)
                -> ExplorationResult
    {
        debug_heading!(
            "complete(completed_goal_depth={:?}, minimums={:?})",
            completed_goal_depth, minimums);

        self.stack[completed_goal_depth].link.take_minimums(minimums);

        // Here:
        // - `table` is the index of the table we are processing
        // - `dfn` is the depth-first number of the associated goal
        // - `link` summarizes the depth-first numbers of the goals that we transitively depend on
        let StackEntry { table: _, link, dfn } = self.stack[completed_goal_depth];

        if link.positive == dfn && link.negative == DepthFirstNumber::MAX {
            self.complete_pop(completed_goal_depth, minimums)
        } else if link.positive == dfn && link.negative >= dfn {
            self.complete_delay(completed_goal_depth, minimums)
        } else {
            Ok(FullyExplored)
        }
    }

    /// Code to cover the sub-case of `complete` in which all of the
    /// goals that we depend upon lie atop us in the stack. Since all
    /// of *them* are completely evaluated, we are done. For example,
    /// imagine that we have this prolog program:
    ///
    /// ```notrust
    /// s :- p.
    /// p :- s.
    /// ```
    ///
    /// We would first push `s` onto the stack of goals with DFN 0,
    /// then `p` with DFN 1. When we finish exploring `p`, we would
    /// invoke `complete`, but it would have a *positive link* on
    /// `s`. This means that the `link.positive` for `q` would be
    /// 0, which is higher than `q`'s DFN of 1. Therefore, we would do nothing.
    /// But then we would invoke `complete` on `s` -- and the link for `s` is 0
    /// as is its DFN. In that case, this `if` is true.
    fn complete_pop(&mut self,
                    completed_goal_depth: StackIndex,
                    minimums: &mut Minimums)
                    -> ExplorationResult {
        debug!(
            "complete_pop(completed_goal_depth={:?}, minimums={:?}",
            completed_goal_depth, minimums);

        let completed_dfn = self.stack[completed_goal_depth].dfn;
        let popped_goals = self.stack.pop_until(completed_goal_depth);
        let mut new_clauses = vec![];
        for popped_goal in popped_goals.into_iter().rev() {
            let StackEntry {
                table: popped_table,
                link: popped_link,
                ..
            } = popped_goal;

            // None of the goals we pop should depend on anything
            // below the completed goal on the stack.
            assert!(popped_link.positive >= completed_dfn,
                    "popped table {:?} with position link {:?} where completed_dfn = {:?}",
                    popped_table,
                    popped_link.positive,
                    completed_dfn);

            // None of the popped goals should have negative dependencies.
            assert!(popped_link.negative == DepthFirstNumber::MAX,
                    "popped table {:?} with negative link {:?} where completed_dfn = {:?}",
                    popped_table,
                    popped_link.negative,
                    completed_dfn);

            // Take the list of negative goals. We will be updating those.
            let negatives = self.tables[popped_table].mark_complete();

            if self.tables[popped_table].answers.is_empty() {
                // If the set of answers is EMPTY, then this goal has
                // definitely FAILED. We can therefore go through the
                // list of clauses blocked negatively on this popped goal
                // and let them proceed.
                let unblocked =
                    negatives
                    .into_iter()
                    .map(|pending| {
                        let PendingExClause {
                            goal_depth,
                            table_goal,
                            selected_goal,
                            delayed_literals,
                            subgoals,
                        } = self.infer.instantiate_canonical(&pending);
                        mem::drop(selected_goal); // don't need it
                        (goal_depth, ExClause {
                            table_goal,
                            delayed_literals,
                            subgoals,
                        })
                    });
                new_clauses.extend(unblocked);
            } else if self.tables[popped_table].is_satisfiable() {
                // We have a definitive answer. We can just
                // abandon the list of negative clauses then,
                // since they are all failed paths.
            } else {
                // Otherwise, if we do not have a *definitive*
                // answer, then it is not yet known whether this
                // goal has succeeded or failed.  We must therefore
                // go through the list of clauses blocked
                // negatively on this popped goal and convert them
                // into DELAYED clauses.
                let delayed = Self::delay(&mut self.infer, negatives);
                new_clauses.extend(delayed);
            }
        }

        *minimums = Minimums {
            positive: DepthFirstNumber::MAX,
            negative: DepthFirstNumber::MAX,
        };
        for (goal_depth, ex_clause) in new_clauses {
            self.new_clause(goal_depth, ex_clause, minimums)?;
        }
        Ok(FullyExplored)
    }

    fn complete_delay(&mut self,
                      completed_goal_depth: StackIndex,
                      minimums: &mut Minimums)
                      -> ExplorationResult {
        debug!(
            "complete_delay(completed_goal_depth={:?}, minimums={:?}",
            completed_goal_depth, minimums);

        let mut new_clauses;

        let top = self.stack.next_index();

        {
            let subgoals = self.stack.peek_until(completed_goal_depth);
            let tables = &mut self.tables;
            let len = subgoals.iter().map(|g| tables[g.table].negatives.len()).sum();
            new_clauses = Vec::with_capacity(len);
            for subgoal in subgoals {
                // Take everything that depends on `subgoal` and convert those
                // depencies into delayed literals. In other words, if `subgoal` is `p`,
                // and we have some negative links arising from something like
                //
                // ```notrust
                // q :- ~p, r
                // ```
                //
                // we would remove the negative link and convert into
                // `q :- ~p | r`.

                let subtable = subgoal.table;
                let negatives = tables[subtable].take_negatives();
                subgoal.link.negative = DepthFirstNumber::MAX;
                new_clauses.extend(Self::delay(&mut self.infer, negatives));
            }
        }

        minimums.positive = self.stack[completed_goal_depth].dfn;
        minimums.negative = DepthFirstNumber::MAX;
        for (goal_depth, ex_clause) in new_clauses {
            self.new_clause(goal_depth, ex_clause, minimums)?;
        }

        // We've again completed all work on the things that were on
        // top of the stack. So `complete` them recursively.
        for table in {
            (completed_goal_depth.value .. top.value)
                .map(|value| StackIndex { value })
                .rev()
        } {
            self.complete(table, minimums)?;
        }

        Ok(FullyExplored)
    }

    fn delay<'i>(infer: &'i mut InferenceTable,
                 negatives: Vec<CanonicalPendingExClause>)
                 -> impl Iterator<Item = (StackIndex, ExClause)> + 'i
    {
        negatives
            .into_iter()
            .map(move |pending| {
                let PendingExClause {
                    goal_depth,
                    table_goal,
                    selected_goal,
                    mut delayed_literals,
                    subgoals,
                } = infer.instantiate_canonical(&pending);

                // delay the selected goal
                delayed_literals.push(selected_goal);

                (goal_depth, ExClause {
                    table_goal,
                    delayed_literals,
                    subgoals,
                })
            })
    }

    ///////////////////////////////////////////////////////////////////////////
    // SLG RESOLVENTS
    //
    // The "SLG Resolvent" is used to combine a *goal* G with some
    // clause or answer *C*.  It unifies the goal's selected literal
    // with the clause and then inserts the clause's conditions into
    // the goal's list of things to prove, basically. Although this is
    // one operation in EWFS, we have specialized variants for merging
    // a program clause and an answer (though they share some code in
    // common).
    //
    // From EWFS:
    //
    // Let G be an X-clause A :- D | L1,...Ln, where N > 0, and Li be selected atom.
    //
    // Let C be an X-clause with no delayed literals. Let
    //
    //     C' = A' :- L'1...L'm
    //
    // be a variant of C such that G and C' have no variables in
    // common.
    //
    // Let Li and A' be unified with MGU S.
    //
    // Then:
    //
    //     S(A :- D | L1...Li-1, L1'...L'm, Li+1...Ln)
    //
    // is the SLG resolvent of G with C.

    /// Applies the SLG resolvent algorithm to incorporate a new
    /// answer and apply it to a previously blocked ex-clause.
    fn resolvent_pending(infer: &mut InferenceTable,
                         pending_ex_clause: &CanonicalPendingExClause,
                         answer_goal: &CanonicalGoal)
                         -> Satisfiable<(StackIndex, ExClause)>
    {
        let PendingExClause {
            goal_depth,
            table_goal,
            selected_goal,
            delayed_literals,
            subgoals,
        } = infer.instantiate_canonical(pending_ex_clause);

        let ex_clause = ExClause {
            table_goal,
            delayed_literals,
            subgoals,
        };

        Self::resolvent_answer(infer, &ex_clause, &selected_goal, answer_goal)
            .map(|r| (goal_depth, r))
    }

    /// Applies the SLG resolvent algorithm to incorporate an answer
    /// produced by the selected literal into the main X-clause,
    /// producing a new X-clause that must be solved.
    ///
    /// # Parameters
    ///
    /// - `ex_clause` is the X-clause we are trying to prove,
    ///   with the selected literal removed from its list of subgoals
    /// - `selected_goal` is the selected literal that was removed
    /// - `answer` is some answer to `selected_goal` that has been found
    fn resolvent_answer(infer: &mut InferenceTable,
                        ex_clause: &ExClause,
                        selected_goal: &InEnvironment<Goal>,
                        answer_goal: &CanonicalGoal)
                        -> Satisfiable<ExClause>
    {
        // Relating the above describes to our parameters:
        //
        // - the goal G is `ex_clause` is, with `selected_goal` being
        //   the selected literal Li, already removed from the list.
        // - the clause C is `answer.` (i.e., the answer has no conditions).

        let result = {
            // C' is now `answer`. No variables in commmon with G.
            let answer_goal = infer.instantiate_canonical(&answer_goal);

            // Perform the SLG resolvent unification.
            Self::resolvent_unify(infer,
                                  ex_clause.clone(),
                                  selected_goal,
                                  &answer_goal,
                                  vec![])
        };

        result
    }

    /// Applies the SLG resolvent algorithm to incorporate a program
    /// clause into the main X-clause, producing a new X-clause that
    /// must be solved.
    ///
    /// # Parameters
    ///
    /// - `goal` is the goal G that we are trying to solve
    /// - `clause` is the program clause that may be useful to that end
    fn resolvent_clause(infer: &mut InferenceTable,
                        goal: &InEnvironment<DomainGoal>,
                        clause: &Binders<ProgramClauseImplication>)
                        -> Satisfiable<ExClause>
    {
        // Relating the above description to our situation:
        //
        // - `goal` G, except with binders for any existential variables.
        //   - Also, we always select the first literal in `ex_clause.literals`, so `i` is 0.
        // - `clause` is C, except with binders for any existential variables.

        // Goal here is now G.
        let ex_clause = ExClause {
            table_goal: goal.clone().cast(),
            delayed_literals: vec![],
            subgoals: vec![],
        };

        // The selected literal for us will always be the main goal
        // `G`. See if we can unify that with C'.
        let environment = &goal.environment;

        // C' in the description above is `consequence :- conditions`.
        //
        // Note that G and C' have no variables in common.
        let ProgramClauseImplication { consequence, conditions } =
            infer.instantiate_binders_in(environment.universe, clause);
        let consequence: InEnvironment<DomainGoal> =
            InEnvironment::new(&environment, consequence);

        Self::resolvent_unify(infer,
                              ex_clause,
                              &goal,
                              &consequence,
                              conditions)
    }

    /// Given the goal G (`goal`) with selected literal Li
    /// (`selected_goal`), the goal environment `environment`, and
    /// the clause C' (`consequence :- conditions`), applies the SLG
    /// resolvent algorithm to yield a new `ExClause`.
    fn resolvent_unify<G>(infer: &mut InferenceTable,
                          mut goal: ExClause,
                          selected_goal: &InEnvironment<G>,
                          consequence: &InEnvironment<G>,
                          conditions: Vec<Goal>)
                          -> Satisfiable<ExClause>
        where G: Zip
    {
        let environment = &selected_goal.environment;

        // Unify the selected literal Li with C'.
        let UnificationResult { goals, constraints } = {
            match infer.unify(&selected_goal.environment, selected_goal, consequence) {
                Err(_) => return Satisfiable::No,
                Ok(v) => v,
            }
        };

        // One (minor) complication: unification for us sometimes yields further domain goals.
        assert!(constraints.is_empty(), "Not yet implemented: region constraints");
        goal.subgoals.extend(goals.into_iter().casted().map(Literal::Positive));

        // Add the `conditions` into the result. One complication is
        // that these are HH-clauses, so we have to simplify into
        // literals first. This can product a sum-of-products. This is
        // why we return a vector.
        goal.subgoals.extend(
            conditions.into_iter()
                      .map(|c| Literal::Positive(InEnvironment::new(&environment, c))));

        Satisfiable::Yes(goal)
    }

    /// Simplifies an HH goal into a series of positive domain goals
    /// and negative HH goals. This operation may fail if the HH goal
    /// includes unifications that cannot be completed.
    fn simplify_hh_goal(infer: &mut InferenceTable,
                        initial_goal: &InEnvironment<Goal>)
                        -> Satisfiable<Product>
    {
        // The positive literals we have accumulated thus far.
        let mut literals = vec![];

        // A stack of higher-level goals to process.
        let mut pending_goals = vec![initial_goal.clone()];

        while let Some(InEnvironment { environment, goal }) = pending_goals.pop() {
            match goal {
                Goal::Quantified(QuantifierKind::ForAll, subgoal) => {
                    let InEnvironment { environment, goal } =
                        subgoal.instantiate_universally(&environment);
                    pending_goals.push(InEnvironment::new(&environment, *goal));
                }
                Goal::Quantified(QuantifierKind::Exists, subgoal) => {
                    let subgoal = infer.instantiate_in(environment.universe,
                                                       subgoal.binders.iter().cloned(),
                                                       &subgoal.value);
                    pending_goals.push(InEnvironment::new(&environment, *subgoal))
                }
                Goal::Implies(wc, subgoal) => {
                    let new_environment = &environment.add_clauses(wc);
                    pending_goals.push(InEnvironment::new(&new_environment, *subgoal));
                }
                Goal::And(subgoal1, subgoal2) => {
                    pending_goals.push(InEnvironment::new(&environment, *subgoal1));
                    pending_goals.push(InEnvironment::new(&environment, *subgoal2));
                }
                Goal::Not(subgoal) => {
                    let subgoal = (*subgoal).clone();
                    literals.push(Literal::Negative(InEnvironment::new(&environment, subgoal)));
                }
                Goal::Leaf(LeafGoal::EqGoal(ref eq_goal)) => {
                    let UnificationResult { goals, constraints } = {
                        match infer.unify(&environment, &eq_goal.a, &eq_goal.b) {
                            Ok(v) => v,
                            Err(_) => return Satisfiable::No,
                        }
                    };

                    assert!(constraints.is_empty(), "Not yet implemented: region constraints");

                    literals.extend(goals.into_iter().casted().map(Literal::Positive));
                }
                Goal::Leaf(LeafGoal::DomainGoal(domain_goal)) => {
                    let domain_goal = domain_goal.cast();
                    literals.push(Literal::Positive(InEnvironment::new(&environment, domain_goal)));
                }
                Goal::CannotProve(()) => {
                    // You can think of `CannotProve` as a special
                    // goal that is only provable if `not {
                    // CannotProve }`. Trying to prove this, of
                    // course, will always create a negative cycle and
                    // hence a delayed literal that cannot be
                    // resolved.
                    literals.push(Literal::Negative(InEnvironment::new(&environment, goal)));
                }
            }
        }

        Satisfiable::Yes(Product { literals })
    }

    ///////////////////////////////////////////////////////////////////////////
    // SLG FACTORS
    //
    // The "SLG Factor" is used to combine a *goal* G with some answer
    // *C*, where C contains delayed literals. It unifies the goal's
    // selected literal with the answer and then inserts the delayed
    // literals into the goal's list of delayed literals.
    //
    // From EWFS:
    //
    // Let G be an X-clause A :- D | L1,...Ln, where N > 0, and Li be selected atom.
    //
    // Let C be an X-clause with delayed literals. Let
    //
    //     C' = A' :- D' |
    //
    // be a variant of C such that G and C' have no variables in
    // common.
    //
    // Let Li and A' be unified with MGU S.
    //
    // Then:
    //
    //     S(A :- D,Li | L1...Li-1, Li+1...Ln)
    //                             ^ see below
    //
    // is the SLG factor of G with C. We alter the process mildly to insert
    // some clauses into `^` -- in particular, the side-effects of unification.

    fn factor_pending(infer: &mut InferenceTable,
                      pending_ex_clause: &CanonicalPendingExClause,
                      answer_goal: &CanonicalGoal)
                      -> Satisfiable<(StackIndex, ExClause)>
    {
        let PendingExClause {
            goal_depth,
            table_goal,
            selected_goal,
            delayed_literals,
            subgoals,
        } = infer.instantiate_canonical(pending_ex_clause);

        let ex_clause = ExClause {
            table_goal,
            delayed_literals,
            subgoals,
        };

        Self::factor(infer, &ex_clause, &selected_goal, answer_goal)
            .map(|c| (goal_depth, c))
    }

    fn factor(infer: &mut InferenceTable,
              ex_clause: &ExClause,
              selected_goal: &InEnvironment<Goal>,
              answer_goal: &CanonicalGoal)
              -> Satisfiable<ExClause>
    {
        let mut ex_clause = ex_clause.clone();

        // C' is now `answer`. No variables in commmon with G.
        let answer_goal = infer.instantiate_canonical(&answer_goal);

        // Unify the selected literal Li with C'.
        let UnificationResult { goals, constraints } = {
            match infer.unify(&selected_goal.environment, selected_goal, &answer_goal) {
                Err(_) => return Satisfiable::No,
                Ok(v) => v,
            }
        };

        // Push Li into the list of delayed literals.
        ex_clause.delayed_literals.push(answer_goal);

        // One (minor) complication: unification for us sometimes yields further domain goals.
        assert!(constraints.is_empty(), "Not yet implemented: region constraints");
        ex_clause.subgoals.extend(goals.into_iter().casted().map(Literal::Positive));

        Satisfiable::Yes(ex_clause)
    }

    ///////////////////////////////////////////////////////////////////////////
    // OVERFLOW
    //
    // An "overflow answer" `X :- <overflow>` indicates that the goal
    // X could not be fully evaluated because it overflowed -- it may
    // be true, false, or unknown. This result propagates winds up
    // propagating upwards to all things that have positive
    // dependencies on X (things with negative dependencies will wind
    // up indefinitely delayed).
    //
    // The "SLG Overflow" is something we invented, analogous to SLG
    // Resolvent and SLG Factor, to describe how an overflow result
    // for some subgoal is combined with the overall goal to yield a
    // new answer. It however works rather differently from those
    // measures.  In particular, the most *analagous* thing to do
    // would be to take the overflow answer and unify it with the
    // selected goal for a pending result. That approach, however,
    // leads to infinite sets of answers in overflow scenarios.
    //
    // So our definition instead is like this. Given a (pending) X-clause
    //
    //     // A :- L0..Li-1, Li, Li+1..Ln
    //
    // and an overflow answer
    //
    //     // C
    //
    // where:
    //
    // - C and Li have no variables in common; and
    // - Li and C unify with subst S
    //
    // we yield the overflow answer A (note: not `S(A)`).
    //
    // Let's work through how overflow propoagates with an example,
    // which will also demonstrate why we don't yield up `S(A)`:
    //
    //     // Foo(S<?A>) :- Foo(?A), WF(Foo(?A)).
    //
    // and the query `Foo(?T)`. This will wind up with a table like:
    //
    //     // Foo(?T) :- Foo(?T)
    //     //        |
    //     // Foo(S<?U>) :- Foo(?U), WF(Foo(?U)).
    //
    // processing the subgoal `Foo(?U)` will create a positive,
    // recursive link (1). The pending ex-clause would be:
    //
    //     // forall<?U> { Foo(S<?U>) :- Foo(?U), WF(Foo(?U)) }
    //
    // where `Foo(?U)` is the selected literal.
    //
    // Next we process `WF(Foo(?U))` -- that will push a new table:
    //
    //     // WF(Foo(?U)) :- WF(Foo(?U))
    //
    // with a positive link (2) to:
    //
    //     // forall<?U> { Foo(S<?U>) :- WF(Foo(?U)) }
    //
    // Now -- let's say -- we trip the overflow counter. Then we will
    // produce an overflow answer `WF(Foo(?U)) :- overflow`. This is
    // propagated along the positive link (2) to produce:
    //
    //     // Foo(S<?U>) :- overflow
    //
    // Now we go to propagate this result along positive link
    // (1). This is where it gets interesting. If we instantiate and
    // unify the selected literal `Foo(?V)` with the overflow answer
    // `Foo(S<?U>)`, we get the substitution:
    //
    //     // S = [?V => S<?U>]
    //
    // If we apply that to the overflow answer `A = Foo(S<?V>)`, we would get
    // `Foo(S<S<?U>>)` -- and you can see then that it is growing.
    //
    // Instead, we do *not* apply the substitution, and so we get
    // `Foo(S<?V>)`.

    fn overflow_pending(infer: &mut InferenceTable,
                        pending_ex_clause: &CanonicalPendingExClause,
                        answer_goal: &CanonicalGoal)
                        -> Satisfiable<(StackIndex, InEnvironment<Goal>)>
    {
        let PendingExClause {
            goal_depth,
            table_goal,
            selected_goal, // <-- Li
            delayed_literals,
            subgoals,
        } = infer.instantiate_canonical(pending_ex_clause);

        let ex_clause = ExClause {
            table_goal,
            delayed_literals,
            subgoals,
        };

        Self::overflow(infer, &ex_clause, &selected_goal, answer_goal)
            .map(|a| (goal_depth, a))
    }

    fn overflow(infer: &mut InferenceTable,
                ex_clause: &ExClause,
                selected_goal: &InEnvironment<Goal>,
                overflow_answer: &CanonicalGoal)
                -> Satisfiable<InEnvironment<Goal>>
    {
        debug!("overflow(ex_clause={:?}, selected_goal={:?}, overflow_answer={:?})",
               ex_clause, selected_goal, overflow_answer);

        let snapshot = infer.snapshot();

        {
            // Once instantiated, `overflow_answer` is C in the description
            // above. No variables in commmon with Li.
            let overflow_answer = infer.instantiate_canonical(&overflow_answer);

            // Unify the selected literal Li with C.
            let UnificationResult { goals, constraints } = {
                match infer.unify(&selected_goal.environment, selected_goal, &overflow_answer) {
                    Err(_) => return Satisfiable::No,
                    Ok(v) => v,
                }
            };

            // We are producing an OVERFLOW ANSWER, which already may or
            // may not hold, so we don't really care about the subgoals
            // and things.
            mem::drop(constraints);
            mem::drop(goals);
        }

        infer.rollback_to(snapshot);

        Satisfiable::Yes(ex_clause.table_goal.clone())
    }
}

impl Stack {
    fn len(&self) -> usize {
        self.stack.len()
    }

    fn next_index(&self) -> StackIndex {
        StackIndex { value: self.stack.len() }
    }

    // Pop all stack frames up to and and including the one at `depth`.
    fn pop_until(&mut self, depth: StackIndex) -> Vec<StackEntry> {
        assert!(depth.value < self.stack.len()); // `depth` must not yet have been popped
        let mut result = Vec::with_capacity(self.stack.len() - depth.value);
        while self.next_index() != depth {
            result.push(self.stack.pop().unwrap());
        }
        result
    }

    fn peek_until(&mut self, depth: StackIndex) -> &mut [StackEntry] {
        &mut self.stack[depth.value .. ]
    }

    fn push(&mut self, table: TableIndex, dfn: DepthFirstNumber) {
        self.stack.push(StackEntry {
            table,
            dfn,
            link: Minimums {
                positive: dfn,
                negative: DepthFirstNumber::MAX,
            }
        });
    }

    fn top(&self) -> &StackEntry {
        self.stack.last().unwrap()
    }
}

impl Index<StackIndex> for Stack {
    type Output = StackEntry;

    fn index(&self, index: StackIndex) -> &StackEntry {
        &self.stack[index.value]
    }
}

impl IndexMut<StackIndex> for Stack {
    fn index_mut(&mut self, index: StackIndex) -> &mut StackEntry {
        &mut self.stack[index.value]
    }
}

impl Tables {
    fn len(&self) -> usize {
        self.tables.len()
    }

    fn next_index(&self) -> TableIndex {
        TableIndex { value: self.tables.len() }
    }

    fn insert(&mut self, goal: &CanonicalGoal, depth: StackIndex) -> TableIndex {
        let index = self.next_index();
        self.tables.push(Table {
            answers: HashMap::new(),
            overflow_answers: HashSet::new(),
            positives: vec![],
            negatives: vec![],
            depth: Some(depth),
        });
        self.table_indices.insert(goal.clone(), index);
        index
    }

    fn index_of(&self, literal: &CanonicalGoal) -> Option<TableIndex> {
        self.table_indices.get(literal).cloned()
    }
}

impl Index<TableIndex> for Tables {
    type Output = Table;

    fn index(&self, index: TableIndex) -> &Table {
        &self.tables[index.value]
    }
}

impl IndexMut<TableIndex> for Tables {
    fn index_mut(&mut self, index: TableIndex) -> &mut Table {
        &mut self.tables[index.value]
    }
}

impl<'a> IntoIterator for &'a mut Tables {
    type IntoIter = <&'a mut Vec<Table> as IntoIterator>::IntoIter;
    type Item = <&'a mut Vec<Table> as IntoIterator>::Item;

    fn into_iter(self) -> Self::IntoIter {
        IntoIterator::into_iter(&mut self.tables)
    }
}

impl Table {
    /// Counts the total number of answers in this table. Note that a
    /// single entry in the `answers` map may contain multiple
    /// answers. This is used for overflow detection to see if there
    /// are too many answers in a particular table.
    fn count_answers(&self) -> usize {
        self.answers
            .iter()
            .map(|(_, delayed_literals_set)| delayed_literals_set.len())
            .sum()
    }

    /// Marks this table as completely evaluated. In the process,
    /// returns the list of pending negative clauses, since those can
    /// possibly now be updated (either to mark them as SUCCESSFUL or
    /// FAILED).
    fn mark_complete(&mut self) -> Vec<CanonicalPendingExClause> {
        let negatives = self.take_negatives();
        self.positives = vec![];
        self.depth = None;
        negatives
    }

    /// Takes the list of negatives and replaces it with an empty
    /// list.  Used when either marking a table as completed or else
    /// delaying a bunch of literals.
    fn take_negatives(&mut self) -> Vec<CanonicalPendingExClause> {
        mem::replace(&mut self.negatives, vec![])
    }

    /// True if this table has at least one solution without a delayed
    /// literal.
    fn is_satisfiable(&self) -> bool {
        self.overflow_answers.is_empty() &&
            self.answers
                .values()
                .any(|delayed_literals| delayed_literals.is_empty())
    }

    /// True if all possible solutions have been evaluated.
    fn is_completely_evaluated(&self) -> bool {
        self.depth.is_none()
    }

    /// Flatten out the answers stored in this table into a set of
    /// (GOAL, DELAYED-LITERAL) pairs.
    fn flat_answers(&self) -> Vec<Answer> {
        let true_answers =
            self.answers
                .iter()
                .flat_map(|(goal, delayed_literals)| {
                    delayed_literals.clone()
                                    .into_set()
                                    .into_iter()
                                    .map(move |dl| Answer {
                                        goal: goal.clone(),
                                        overflow: false,
                                        delayed_literals: dl.delayed_literals.clone(),
                                    })
                });

        let overflow_answers =
            self.overflow_answers
                .iter()
                .cloned()
                .map(|goal| Answer { goal, overflow: true, delayed_literals: vec![] });

        true_answers.chain(overflow_answers).collect()
    }
}

impl DelayedLiteralsSet {
    /// Number of distinct `DelayedLiterals` values represented here.
    fn len(&self) -> usize {
        match *self {
            DelayedLiteralsSet::None => 1,
            DelayedLiteralsSet::Some(ref set) => set.len(),
        }
    }

    fn is_empty(&self) -> bool {
        match *self {
            DelayedLiteralsSet::None => true,
            DelayedLiteralsSet::Some(_) => false,
        }
    }

    fn into_set(self) -> HashSet<DelayedLiterals> {
        match self {
            DelayedLiteralsSet::None => {
                let mut set = HashSet::new();
                set.insert(DelayedLiterals { delayed_literals: vec![] });
                set
            }
            DelayedLiteralsSet::Some(s) => s,
        }
    }
}

impl Minimums {
    /// Update our fields to be the minimum of our current value
    /// and the values from other.
    fn take_minimums(&mut self, other: &Minimums) {
        self.positive = min(self.positive, other.positive);
        self.negative = min(self.negative, other.negative);
    }

    fn take_negative_minimum(&mut self, other: DepthFirstNumber) {
        self.negative = min(self.negative, other);
    }

    fn minimum_of_pos_and_neg(&self) -> DepthFirstNumber {
        min(self.positive, self.negative)
    }

    fn reset_to_bottom_of_stack(&mut self) {
        self.positive = DepthFirstNumber::MIN;
        self.negative = DepthFirstNumber::MIN;
    }
}

impl DepthFirstNumber {
    const MIN: DepthFirstNumber = DepthFirstNumber { value: 0 };
    const MAX: DepthFirstNumber = DepthFirstNumber { value: ::std::u64::MAX };

    fn next(&mut self) -> DepthFirstNumber {
        let value = self.value;
        assert!(value < ::std::u64::MAX);
        self.value += 1;
        DepthFirstNumber { value }
    }
}

impl TableIndex {
    /// Index of the root table. This is always the first table
    /// created.
    const ROOT: TableIndex = TableIndex { value: 0 };
}

impl ExClause {
    fn with_subgoals(mut self, literals: Vec<Literal>) -> Self {
        self.subgoals.extend(literals);
        self
    }
}

impl<T> Satisfiable<T> {
    fn yes(self) -> Option<T> {
        match self {
            Satisfiable::Yes(v) => Some(v),
            Satisfiable::No => None,
        }
    }

    fn map<F, U>(self, op: F) -> Satisfiable<U>
        where F: FnOnce(T) -> U,
    {
        match self {
            Satisfiable::Yes(v) => Satisfiable::Yes(op(v)),
            Satisfiable::No => Satisfiable::No,
        }
    }
}

impl<'a> TruthValues<'a> {
    fn new(tables: &'a Tables) -> Self {
        let truth_values = vec![None; tables.len()];
        TruthValues { tables, truth_values }
    }

    fn simplified_answers(&mut self, table_index: TableIndex) -> Vec<Answer> {
        let mut answers = self.tables[table_index].flat_answers();

        // First, weed out the answers that are false.
        answers.retain(|answer| match self.eval_answer(answer) {
            TruthValue::False => false,
            _ => true,
        });

        // Next, simplify the answers that remain. Weed out delayed
        // literals that can be evaluated to definite values.
        for answer in &mut answers {
            answer.delayed_literals.retain(|literal| match self.eval_goal(&literal) {
                TruthValue::False =>
                    false,

                TruthValue::Unknown =>
                    true,

                // we should have screen out this case just above, when we weeded
                // out answers that are false
                TruthValue::True =>
                    panic!("delayed literal {:?} is true but answer is true", literal),
            });
        }

        // Finally, sort and prune out duplicates.
        answers.sort();
        answers.dedup();

        answers
    }

    fn eval_goal(&mut self, goal: &CanonicalGoal) -> TruthValue {
        let table = self.tables.index_of(goal).unwrap();
        self.eval_table(table)
    }

    fn eval_table(&mut self, table: TableIndex) -> TruthValue {
        if let Some(truth_value) = self.truth_values[table.value] {
            return truth_value;
        }

        // Set initial value, before we begin our walk. If we see cycles,
        // they are considered unknown results.
        self.truth_values[table.value] = Some(TruthValue::Unknown);

        // Compute result. A table is true if any of its answers are true.
        let truth_value =
            TruthValue::any(
                self.tables[table]
                    .flat_answers()
                    .into_iter()
                    .map(|answer| self.eval_answer(&answer)));

        // Update table.
        self.truth_values[table.value] = Some(truth_value);

        truth_value
    }

    fn eval_answer(&mut self, answer: &Answer) -> TruthValue
    {
        let is_overflow = if answer.overflow {
            TruthValue::Unknown
        } else {
            TruthValue::True
        };

        TruthValue::all(
            answer.delayed_literals
                .iter()
                .map(|dl| self.eval_goal(dl).not()))
            .and(is_overflow)
    }
}

impl TruthValue {
    fn or(self, other: TruthValue) -> TruthValue {
        match (self, other) {
            (TruthValue::True, _) | (_, TruthValue::True) => TruthValue::True,
            (TruthValue::Unknown, _) | (_, TruthValue::Unknown) => TruthValue::Unknown,
            (TruthValue::False, TruthValue::False) => TruthValue::False,
        }
    }

    fn and(self, other: TruthValue) -> TruthValue {
        match (self, other) {
            (TruthValue::False, _) | (_, TruthValue::False) => TruthValue::False,
            (TruthValue::Unknown, _) | (_, TruthValue::Unknown) => TruthValue::Unknown,
            (TruthValue::True, TruthValue::True) => TruthValue::True,
        }
    }

    fn not(self) -> TruthValue {
        match self {
            TruthValue::True => TruthValue::False,
            TruthValue::False => TruthValue::True,
            TruthValue::Unknown => TruthValue::Unknown,
        }
    }

    fn any<I>(iter: I) -> TruthValue
        where I: IntoIterator<Item = TruthValue>
    {
        iter.into_iter().fold(TruthValue::False, |a, b| a.or(b))
    }

    fn all<I>(iter: I) -> TruthValue
        where I: IntoIterator<Item = TruthValue>
    {
        iter.into_iter().fold(TruthValue::True, |a, b| a.and(b))
    }
}
