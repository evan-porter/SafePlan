import yaml
import sqlite3
import csv
import contextlib
import os
import pickle
from glob import glob
from datetime import datetime


def add_rows(db_name: str, table: str, rows: list) -> None:
    """Adds an arbitrary number of rows to the database as a batch

    Args:
        db_name (str): Database name to add rows
        table (str): Table in the database to add the rows
        rows (list): List of lists in format rows, columns of data

    Raises:
        sqlite3.OperationalError: Raises an operational error if formatting is incorrect
    """
    with contextlib.closing(make_conn(db_name)) as conn:
        with conn:
            if isinstance(rows, dict):
                rows = [rows]
            for i, r in enumerate(rows):
                if isinstance(r, dict):
                    rows[i] = list(r.values())

            if len(rows):
                qmarks = "?" + ",?" * (len(rows[0]) - 1)
                try:
                    query = f'INSERT OR IGNORE INTO {table} VALUES ({qmarks})'
                    conn.executemany(query, rows)
                except Exception:
                    print(f'Could not add rows to {table}: {rows}')
                    conn.executemany(query, rows)
                    raise sqlite3.OperationalError


def read_yaml(filename: str):
    with open(filename, 'r') as yf:
        return yaml.safe_load(yf)


def read_pickle(filename: str):
    objects = []
    with (open(filename, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects


def parse_pickle_file(filename):
    object_dict = read_pickle(filename)[0]
    structures = list(object_dict['empty_structures'].keys())
    for settings in object_dict['structure_settings'].values():
        for template in settings['template_specific_config'].values():
            structures.append(template['name'])
    return list(set(structures))


def dbs_from_scratch():
    """Creates the reference and logging databases from the config .yaml files
    """
    timestamp = isotime()
    root = os.path.realpath('..')
    ref_db_name = os.path.join(root, 'dbs', f'ReferenceConstraints_{timestamp}.db')
    initialize_db(ref_db_name, config_file=os.path.join(root, 'configs', 'reference_config.yaml'))

    # Dose constraints
    doses = read_csv(os.path.join(root, 'constraints', 'updated_constraints.csv'))
    to_add = []
    fixed = []
    for row in doses[1:]:
        if '*' in row[0]:
            to_add.append([row[0].replace('*', 'L'), *row[1:]])
            to_add.append([row[0].replace('*', 'R'), *row[1:]])
        else:
            fixed.append(row)
    fixed.extend(to_add)
    fixed.sort(key= lambda x: x[0])
    add_rows(ref_db_name, 'DoseConstraints', fixed)

    templates = read_csv(os.path.join(root, 'templates', 'SiteTemplates.csv'))
    rois = [x[0] for x in templates if 'XXXX' not in x[0] and x[0] != 'RoiName']
    rois.sort()

    add_rows(ref_db_name, 'CompliantNames', [[x] for x in rois])

    igrts = read_csv(os.path.join(root, 'templates', 'IGRTStructures.csv'))
    add_rows(ref_db_name, 'IGRTStructures', igrts[1:])

    log_db_name = os.path.join(root, 'dbs', f'ChangeLog_{timestamp}.db')
    log_files = glob('../dbs/ChangeLog*.db')
    log_files.sort()

    if len(log_files):
        new_db = sqlite3.connect(log_db_name)
        old_db = sqlite3.connect(log_files[-1])

        # Dump old database in the new one.
        query = "".join(line for line in old_db.iterdump())
        new_db.executescript(query)
    else:
        initialize_db(log_db_name, config_file=os.path.join(root, 'configs', 'logging_config.yaml'))


def dict_factory(cursor: sqlite3.Cursor, row: list) -> dict:
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def initialize_db(db_path: str, config_file: str) -> sqlite3.Connection:
    """Initializes database at db_path as described in tables
    Args:
        config_file (str): A string to the config path.
        db_path (str): Posix path or :memory:
    Returns:
        sqlite3.Connection: Returns a sqlite3 connection to the database
    """
    tables = read_yaml(config_file)
    with make_conn(db_path) as conn:
        for table in tables.keys():
            field_set = [f"'{col}' {fmt}" for col, fmt in tables[table].items()]

            if len(field_set):
                field_fmt = ", ".join(field_set)
                query = f"CREATE TABLE IF NOT EXISTS {table} ({field_fmt})"
                conn.execute(query)
        return conn


def isotime():
    return datetime.now().replace(microsecond=0).astimezone().isoformat()


def make_conn(db_path: str, dicts: bool = False) -> sqlite3.Connection:
    """Makes a connection to the database at db_path
    Args:
        db_path (str): A Posix path or :memory:
    Returns:
        sqlite3.Connection: A connection to the database
    Notes:
        Uses PRAGMA synchronous = Extra, journal_mode = WAL
        Does not check for same thread, allowing for multithreading
    """
    conn = sqlite3.connect(db_path, check_same_thread=False,
                           uri=True, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.execute('''PRAGMA synchronous = FULL''')
    conn.execute('''PRAGMA journal_mode = WAL2''')
    if dicts:
        conn.row_factory = dict_factory
    return conn


def read_csv(filename: str) -> list:
    with open(filename, 'r', encoding='utf-8-sig') as f:
        csvreader = csv.reader(f, delimiter=',', quotechar='|')
        rows = []
        for row in csvreader:
            if not all([x == '' for x in row]):
                rows.append([x.strip() if x else 'NULL' for x in row])
        return rows


if __name__ == '__main__':
    root = os.path.realpath('..')
    dbs_from_scratch()
