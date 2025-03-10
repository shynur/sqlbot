#! python3.13
# -*- coding: utf-8-unix; -*-
import sqlite3

try:
    __import__("os").remove("eg.sqlite3")
except FileNotFoundError:
    pass
print(
    sqlite3.complete_statement("SELECT foo FROM bar;"),
    sqlite3.complete_statement("SELECT foo"),
    "\n",
)

db = sqlite3.connect("eg.sqlite3", autocommit=False)
db.row_factory = (
    lambda cursor, row: __import__("collections")
    .namedtuple("Row", [column[0] for column in cursor.description])
    ._make(row)
)


with db:
    db.execute("CREATE TABLE movie(title, year, score);")

print(db.execute("SELECT name FROM sqlite_master;").fetchone(), "\n")
print(
    db.execute("SELECT name FROM sqlite_master WHERE name='spam';").fetchone() is None,
    "\n",
)

with db:
    db.execute(
        """
        INSERT INTO movie VALUES
            ('Monty Python and the Holy Grail', 1975, 8.2),
            ('And Now for Something Completely Different', 1971, 7.5);
        """
    )

for row in db.execute("SELECT title, score FROM movie;").fetchall():
    print(row)
print("\n")

with db:
    db.executemany(
        "INSERT INTO movie VALUES(?, ?, ?);",
        [
            ("Monty Python Live at the Hollywood Bowl", 1982, 7.9),
            ("Monty Python's The Meaning of Life", 1983, 7.5),
            ("Monty Python's Life of Brian", 1979, 8.0),
        ],
    )
for row in db.execute("SELECT year, title FROM movie ORDER BY year;"):
    print(row)
print("\n")

with db:
    db.execute("CREATE TABLE lang(name, first_appeared)")
    db.executemany(
        "INSERT INTO lang VALUES(:name, :year)",
        [
            {"name": "C", "year": 1972},
            {"name": "Fortran", "year": 1957},
            {"name": "Python", "year": 1991},
            {"name": "Go", "year": 2009},
        ],
    )
print(
    db.execute("SELECT * FROM lang WHERE first_appeared = ?", (1972,)).fetchall(), "\n"
)
