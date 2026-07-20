#the web folder ships copies of things that live elsewhere in the repo,
#because railway only deploys web/. the copies MUST stay identical: if
#clean_line drifts, the line picker silently stops matching page lines to
#their database rows. this asserts they haven't, and the check workflow
#runs it on every push so a drift can never reach a deploy unnoticed.
#run it locally from the repo root the same way:
#    python tools/check_sync.py
#
#functions are compared by ast (comments and blank lines don't count,
#behavior does), the generated prefix_words files byte for byte

import ast
import sys
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
problems = []


def read(path):
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        return f.read()


def func_dump(path, name):
    for node in ast.walk(ast.parse(read(path))):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.dump(node)
    problems.append(path + " has no function " + name)
    return None


def assign_value(path, name):
    for node in ast.parse(read(path)).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    problems.append(path + " has no assignment " + name)
    return None


def same(what, a, b, where="between web/ and its source of truth"):
    if a is not None and b is not None and a != b:
        problems.append(what + " drifted " + where)


#the line cleaner: it must clean exactly like the ingest did, or the line
#picker can't find the lines shown on the page in the lines table. the helper
#and the keyword list it leans on are checked too - clean_line's own ast would
#look identical while either of those quietly said something different
same("clean_line", func_dump("web/app.py", "clean_line"), func_dump("common/cards.py", "clean_line"))
same("reminder_is_the_rule", func_dump("web/app.py", "reminder_is_the_rule"),
     func_dump("common/cards.py", "reminder_is_the_rule"))
same("REMINDER_KEYWORDS", assign_value("web/app.py", "REMINDER_KEYWORDS"),
     assign_value("common/cards.py", "REMINDER_KEYWORDS"))

#the generated scryfall word catalogs the cleaner leans on
if read("web/prefix_words.py") != read("common/prefix_words.py"):
    problems.append("prefix_words.py drifted between web/ and common/")

#the calibration seeds (the database's meta copy wins at runtime, but the
#seeds cover virgin databases and should agree too)
same("CALIBRATION seed", assign_value("web/app.py", "CALIBRATION"), assign_value("common/concept.py", "CALIBRATION"))
same("MECH_CALIBRATION seed", assign_value("web/app.py", "MECH_CALIBRATION"), assign_value("ingest/update.py", "MECH_CALIBRATION"))

#the report bakeoff scores pairs the way the site does, with its own copies
#of the two scoring functions
for fn in ("line_weight", "mech_display"):
    same(fn, func_dump("finetune/pairs_bakeoff.py", fn), func_dump("web/app.py", fn),
         "between finetune/pairs_bakeoff.py and web/app.py")

if problems:
    for p in problems:
        print("DRIFT: " + p)
    sys.exit(1)
print("all web copies match their sources of truth")
