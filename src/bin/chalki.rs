extern crate rustyline;
extern crate chalk_parse;
extern crate chalk;
extern crate docopt;

#[macro_use]
extern crate serde_derive;

#[macro_use]
extern crate error_chain;

use std::io::Read;
use std::fs::File;
use std::sync::Arc;
use std::process::exit;

use chalk::ir;
use chalk::lower::*;
use chalk::solve::solver::{self, Solver, CycleStrategy};
use chalk::solve::slg;
use docopt::Docopt;
use rustyline::error::ReadlineError;

const USAGE: &'static str = "
chalk repl

Usage:
  chalki [options]
  chalki (-h | --help)

Options:
  --help              Show this screen.
  --program=PATH      Specifies the path to the `.chalk` file containing traits/impls.
  --goal=GOAL         Specifies a goal to evaluate (may be given more than once).
  --overflow-depth=N  Specifies the overflow depth (default: 10).
  --slg               Use the experimental SLG resolution system.
";

#[derive(Debug, Deserialize)]
struct Args {
    flag_program: Option<String>,
    flag_goal: Vec<String>,
    flag_overflow_depth: usize,
    flag_slg: bool,
}

error_chain! {
    links {
        Parse(chalk_parse::errors::Error, chalk_parse::errors::ErrorKind);
        Chalk(chalk::errors::Error, chalk::errors::ErrorKind);
    }

    foreign_links {
        Io(::std::io::Error);
        Rustyline(ReadlineError);
    }
}

struct Program {
    text: String,
    ir: Arc<ir::Program>,
    env: Arc<ir::ProgramEnvironment>,
}

impl Program {
    fn new(text: String) -> Result<Program> {
        let ir = Arc::new(chalk_parse::parse_program(&text)?.lower()?);
        let env = Arc::new(ir.environment());
        Ok(Program { text, ir, env })
    }
}

quick_main!(run);

fn run() -> Result<()> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.deserialize())
        .unwrap_or_else(|e| e.exit());

    // Initialize global overflow depth before everything
    solver::set_overflow_depth(args.flag_overflow_depth);

    let mut prog = None;

    if let Some(program) = args.flag_program {
        match load_program(&program) {
            Ok(p) => prog = Some(p),
            Err(err) => {
                println!("error loading program: {}", err);
                exit(1);
            }
        }
    }

    let use_slg = args.flag_slg;
    if args.flag_goal.is_empty() {
        readline_loop(&mut rustyline::Editor::new(), "?- ", |rl, line| {
            if let Err(e) = process(use_slg, line, rl, &mut prog) {
                println!("error: {}", e);
            }
        })
    } else {
        let prog = match prog {
            Some(p) => p,
            None => {
                println!("error: cannot eval with a program, use `--program`");
                exit(1);
            }
        };

        let flag_goal = args.flag_goal;
        ir::set_current_program(&prog.ir, || -> Result<()> {
            for g in flag_goal {
                if let Err(e) = goal(use_slg, &g, &prog) {
                    println!("error: {}", e);
                    exit(1);
                }
            }
            Ok(())
        })?;

        Ok(())
    }
}

/// Repeatedly calls `f`, passing in each line, using the given promt, until EOF is received
fn readline_loop<F>(rl: &mut rustyline::Editor<()>, prompt: &str, mut f: F) -> Result<()>
    where F: FnMut(&mut rustyline::Editor<()>, &str)
{
    loop {
        match rl.readline(prompt) {
            Ok(line) => {
                rl.add_history_entry(&line);
                f(rl, &line);
            }
            Err(ReadlineError::Eof) => break,
            Err(e) => Err(e)?,
        }
    }

    Ok(())
}

/// Process a single command
fn process(use_slg: bool,
           command: &str,
           rl: &mut rustyline::Editor<()>,
           prog: &mut Option<Program>)
           -> Result<()> {
    if command == "help" {
        help()
    } else if command == "program" {
        *prog = Some(Program::new(read_program(rl)?)?);
    } else if command.starts_with("load ") {
        let filename = &command["load ".len()..];
        *prog = Some(load_program(filename)?);
    } else {
        let prog = prog.as_ref().ok_or("no program currently loaded")?;
        ir::set_current_program(&prog.ir, || -> Result<()> {
            match command {
                "print" => println!("{}", prog.text),
                "lowered" => println!("{:#?}", prog.env),
                _ => goal(use_slg, command, prog)?,
            }
            Ok(())
        })?
    }

    Ok(())
}

fn load_program(filename: &str) -> Result<Program> {
    let mut text = String::new();
    File::open(filename)?.read_to_string(&mut text)?;
    Ok(Program::new(text)?)
}

fn help() {
    println!("Commands:");
    println!("  help         print this output");
    println!("  program      provide a program via stdin");
    println!("  load <file>  load program from <file>");
    println!("  print        print the current program");
    println!("  lowered      print the lowered program");
    println!("  <goal>       attempt to solve <goal>");
}

fn read_program(rl: &mut rustyline::Editor<()>) -> Result<String> {
    println!("Enter a program; press Ctrl-D when finished");
    let mut text = String::new();
    readline_loop(rl, "| ", |_, line| {
        text += line;
        text += "\n";
    })?;
    Ok(text)
}

fn goal(use_slg: bool, text: &str, prog: &Program) -> Result<()> {
    let goal = chalk_parse::parse_goal(text)?.lower(&*prog.ir)?;
    if use_slg {
        match slg::solve_repl(solver::get_overflow_depth(), &prog.env, *goal) {
            Ok(answers) => {
                println!("{} answers found: {:#?}\n", answers.answers.len(), answers);
            }
            Err(error) => {
                println!("exploration error: {:?}\n", error);
            }
        }
    } else {
        let mut solver = Solver::new(&prog.env, CycleStrategy::Tabling, solver::get_overflow_depth());
        let goal = ir::InEnvironment::new(&ir::Environment::new(), *goal);
        match solver.solve_closed_goal(goal) {
            Ok(v) => println!("{}\n", v),
            Err(e) => println!("No possible solution: {}\n", e),
        }
    }
    Ok(())
}
