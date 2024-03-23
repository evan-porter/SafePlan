import sqlite3
import difflib
import hashlib
import datetime
import yaml
import os
import contextlib
import re
import ast
import pandas as pd
import numpy as np
import random
import clr
from decimal import Decimal
from connect import CompositeAction
from dataclasses import dataclass


clr.AddReference('System.Drawing')
import System.Drawing


@dataclass
class Status:
    level: str
    text: str


@dataclass
class IssueLeaf:
    beam_name: str
    segment: int
    leaf: int
    bank: str

    def __str__(self):
        return f'Beam {self.beam_name}, segment {self.segment}, leaf {self.leaf}, bank {self.bank}\n'

    def __lt__(self, other):
        if self.beam_name != other.beam_name:
            return self.beam_name < other.beam_name
        elif self.segment != other.segment:
            return self.segment < other.segment
        elif self.leaf != other.leaf:
            return self.leaf < other.leaf
        else:
            return self.bank < other.bank


def get_columns(db_name, table):
    columns_data = query_db(db_name, f'PRAGMA table_info({table})')
    columns = [x[1] for x in columns_data]
    return columns


def isotime():
    time = datetime.datetime.now().replace(microsecond=0).astimezone().isoformat()
    return time.replace(':', '.')


def dict_factory(cursor: sqlite3.Cursor, row: list) -> dict:
    """Converts SQLite queries into dictonary format
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


# Converts np.array to TEXT when inserting
def adapt_array(arr: np.ndarray) -> str:
    """Adds SQLite ARRAY type
    """
    return np.array2string(arr, separator=',', suppress_small=True)


sqlite3.register_adapter(np.ndarray, adapt_array)


# Converts TEXT to np.array when selecting
def convert_array(text: str) -> np.ndarray:
    """Adds SQLite ARRAY type
    """
    decode = text.decode('utf-8')
    pattern = re.compile("^[\[\] 0-9,.-]*$")  # check in basic format
    if pattern.match(decode.replace('\n', '')):
        return np.array(ast.literal_eval(decode)).astype(np.float32)
    return decode


sqlite3.register_converter('ARRAY', convert_array)


def adapt_name_list(names: list) -> str:
    """Adds SQLite NAMELIST type
    """
    if names == 'NULL':
        return 'NULL'
    return ','.join(names)


sqlite3.register_adapter(list, adapt_name_list)


def convert_name_list(text: str) -> str:
    """Adds SQLite NAMELIST type
    """
    if text == b'NULL':
        return text
    return text.decode('utf-8').split(',')


sqlite3.register_converter('NAMELIST', convert_name_list)


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


def initialize_db(db_path: str, config_file: str) -> sqlite3.Connection:
    """Initializes database at db_path as described in tables
    Args:
        config_file (str): A string to the config path.
        db_path (str): Posix path or :memory:
    Returns:
        sqlite3.Connection: Returns a sqlite3 connection to the database
    """
    tables = read_yaml(config_file)
    with make_conn(db_path, True) as conn:
        for table in tables.keys():
            field_set = [f"'{col}' {fmt}" for col,
                         fmt in tables[table].items()]

            if len(field_set):
                field_fmt = ", ".join(field_set)
                query = f"CREATE TABLE IF NOT EXISTS {table} ({field_fmt})"
                conn.execute(query)
        return conn


def read_yaml(filename: str) -> list:
    """Reads a json specified by a filepath
    """
    with open(filename, 'r') as yf:
        contents = yaml.safe_load(yf)
    return contents


def query_db(db_name: str, query: str, dicts: bool = False) -> list:
    """Handles queries to the database cleanly

    Args:
        db_name (str): Database name or path
        query (str): Formatted SQLite3 query
        dicts (bool, optional): Specifies if results are returned as dict (True)
            or as lists (False). Defaults to False.

    Returns:
        list: A list of the query results
    """
    with contextlib.closing(make_conn(db_name, dicts)) as conn:
        with conn:
            return conn.execute(query).fetchall()


def replace_in_db(db_name: str, query: str) -> None:
    """Handles replacement in the database cleanly
    """
    with contextlib.closing(make_conn(db_name)) as conn:
        with conn:
            conn.execute(query)


def add_rows(db_name: str, table: str, rows: list, replace: bool = False,
             reorder: bool = True) -> None:
    """Adds rows to the database

    Args:
        db_name (str): Database name
        table (str): Table in database
        rows (list): List of rows to add to the database
        replace (bool): Specifies if an existing unique value should be overwritten (True)
            or ignored (False). Defaults to False.

    Raises:
        sqlite3.OperationalError: A connection to keep database open if in memory
    """
    with contextlib.closing(make_conn(db_name)) as conn:
        with conn:
            if isinstance(rows, dict):
                if reorder:
                    columns = get_columns(db_name, table)
                    rows = {c: rows[c] for c in columns}
                rows = [rows]
            for i, r in enumerate(rows):
                if isinstance(r, dict):
                    if reorder:
                        columns = get_columns(db_name, table)
                        r = {c: r[c] for c in columns}
                    rows[i] = list(r.values())

            if len(rows):
                qmarks = "?" + ",?" * (len(rows[0]) - 1)
                try:
                    if replace:
                        query = f'INSERT OR REPLACE INTO {table} VALUES ({qmarks})'
                    else:
                        query = f'INSERT OR IGNORE INTO {table} VALUES ({qmarks})'
                    conn.executemany(query, rows)
                except Exception:
                    print(f'Could not add rows to {table}: {rows}')
                    conn.executemany(query, rows)
                    raise sqlite3.OperationalError


class BackendManager:
    def __init__(self, get_current: object, ref_db_path: str, log_db_path: str) -> None:
        self.db_name = 'file::memory:?cache=shared'
        db_dir = os.path.dirname(ref_db_path)
        project_dir = os.path.dirname(db_dir)
        self.conn = initialize_db(db_path=self.db_name,
                                  config_file=os.path.join(project_dir, 'configs', 'patient_config.yaml'))
        self.conn.execute(f'ATTACH DATABASE "{ref_db_path}" AS refdb')
        self.patient = get_current('Patient')
        self.case = get_current('Case')
        self.examination = get_current('Examination')
        try:
            self.plan = get_current('Plan')
        except Exception:
            self.plan = None

        try:
            self.beam_set = get_current('BeamSet')
        except Exception:
            self.beam_set = None

        self.ref_db_path = ref_db_path
        self.log_db_path = log_db_path

        if self.plan is None:
            self.number_of_fx = None
        else:
            try:
                self.number_of_fx = int(self.beam_set.FractionationPattern.NumberOfFractions)
            except Exception:
                self.number_of_fx = None

        # Move these to the functions to run if needed, saves runtime
        self._pull_rois()

        if self.plan is not None:
            self._pull_clinical_goals()

    @property
    def uid(self):
        return f'{self.patient.PatientID}_{self.case.PerPatientUniqueId}_{self.case.CaseName}'

    def _approved(self) -> None:
        """Checks for contour approval
        """
        approved = [x.ApprovedStructureSets for x in self.case.PatientModel.StructureSets]
        return any(approved)

    def _hash_row(self, row) -> list:
        sha = hashlib.sha256()
        for d in row.values():
            sha.update(str.encode(str(d)))
        row['Data UID'] = sha.hexdigest()
        return row

    def _hash_rows(self, rows) -> list:
        return list(map(self._hash_row, rows))

    def _add_df_to_log_db(self, table, df, replace=False, return_df=True):
        rows = df.to_dict('records')
        hash_rows = self._hash_rows(rows)
        add_rows(self.log_db_path, table, hash_rows, replace)
        if return_df:
            iloc = list(df.columns).index('Data UID')
            df['Data UID'] = [r[iloc] for r in hash_rows]
            return df

    def _pull_clinical_goals(self) -> None:
        """Pulls clinical goals from RayStation patient information and organizes into
            patient database for later comparison
        """
        # Order of RoiName, GoalType, GoalTolerance, GoalDose, GoalVolume, GoalVolumeUnits
        goal_rows = []
        for efn in self.plan.TreatmentCourse.EvaluationSetup.EvaluationFunctions:
            goal = efn.PlanningGoal

            row = {}
            row['ROI Name'] = efn.ForRegionOfInterest.Name
            row['Goal Type'] = goal.Type
            row['Goal Tolerance'] = goal.Tolerance
            row['Goal Criteria'] = goal.GoalCriteria

            if 'Absolute' in goal.Type:
                row['Volume Units'] = 'cc'
            elif 'Average' in goal.Type:
                row['Volume Units'] = ''
            else:
                row['Volume Units'] = 'percent'

            if goal.Type == 'VolumeAtDose':
                row['Dose Limit'] = goal.ParameterValue
                row['Volume'] = goal.AcceptanceLevel * 100
            elif goal.Type == 'AbsoluteVolumeAtDose':
                row['Dose Limit'] = goal.ParameterValue
                row['Volume'] = goal.AcceptanceLevel
            elif goal.Type == 'DoseAtVolume':
                row['Dose Limit'] = goal.AcceptanceLevel
                row['Volume'] = goal.ParameterValue * 100
            elif goal.Type == 'DoseAtAbsoluteVolume':
                row['Dose Limit'] = goal.AcceptanceLevel
                row['Volume'] = goal.ParameterValue
            elif goal.Type == 'AverageDose':
                row['Dose Limit'] = goal.AcceptanceLevel
                row['Volume'] = 'mean'
            else:
                break

            goal_rows.append(row)
        add_rows(self.db_name, 'ClinicalGoals', goal_rows)

    def _pull_rois(self) -> None:
        """Pulls ROIs from the patient information
        """
        roi_rows = []
        for structure_set in self.case.PatientModel.StructureSets:
            for roi in structure_set.RoiGeometries:
                row = {}
                row['ROI Name'] = roi.OfRoi.Name
                if roi.HasContours():
                    row['Volume'] = roi.GetRoiVolume()
                else:
                    row['Volume'] = 0
                row['Volume Units'] = 'cc'
                row['ROI Type'] = roi.OfRoi.OrganData.OrganType
                roi_rows.append(row)
        add_rows(self.db_name, 'Rois', roi_rows)

    def has_experimental(self):
        rois = query_db(self.db_name, 'SELECT * FROM Rois', dicts=True)
        for roi in rois:
            if '_experimental' in roi['ROI Name']:
                return True
        return False

    def add_or_edit(self, text, user, table) -> None:
        query = f'SELECT EXISTS(SELECT * FROM {table} WHERE "Plan UID" LIKE "{self.uid}")'
        exists = query_db(self.log_db_path, query)[0][0]
        if exists:
            query = f'SELECT * FROM {table} WHERE "Plan UID" LIKE "{self.uid}"'
            row = query_db(self.log_db_path, query, dicts=True)[0]
            row[user] = text
            add_rows(self.log_db_path, table, row, replace=True)
        else:
            columns = get_columns(self.log_db_path, table)
            row = {k: '' for k in columns}
            row['Plan UID'] = self.uid
            row[user] = text
            add_rows(self.log_db_path, table, row)

    def write_responses(self, responses):
        columns = self.questionnaire()

        row = {'Plan UID': self.plan.PlanId,
               'MRN': self.patient.PatientId,
               'User': os.getlogin(),
               'Plan Name': self.plan.Name,
               }
        for q, v in zip(columns, responses.values()):
            row[q] = v

        add_rows(self.log_db_path, 'Questionnaire', row, replace=True)

    def questionnaire(self):
        return get_columns(self.log_db_name, 'Questionnaire')

    def write_comment(self, text, user):
        self.add_or_edit(text=text, user=user, table='UserComments')

    def record_run(self, run_type):
        self.add_or_edit(text=True, user=run_type, table='ScriptRunRecords')

    def multiple_beamsets(self) -> bool:
        return len(self.plan.BeamSets) > 1

    def record_fractions(self, fractions) -> None:
        row = {'Plan UID': self.uid,
               'Fractions': fractions}
        add_rows(self.log_db_path, 'FractionSpecification', row, replace=True)

    def check_if_run(self, run_type):
        table = 'ScriptRunRecords'
        query = f'SELECT EXISTS(SELECT * FROM {table} WHERE "Plan UID" LIKE "{self.uid}")'
        exists = query_db(self.log_db_path, query)[0][0]

        if not exists:
            return False

        query = f'SELECT {run_type} FROM ScriptRunRecords WHERE "Plan UID" LIKE "{self.uid}"'
        already_run = query_db(self.log_db_path, query)[0][0]
        return already_run

    def renaming_history(self) -> list:
        rois = query_db(self.log_db_path, f'SELECT * FROM RoiRenaming WHERE "Plan UID" LIKE "{self.uid}"', dicts=True)
        df = pd.DataFrame(columns=['Original Name', 'Current Name'])

        for roi in rois:
            if roi['Original Name'] == roi['Current Name']:
                continue
            row = {'Original Name': roi['Original Name'], 'Current Name': roi['Current Name']}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        return df

    def check_reference_limit(self, row: dict, roi: dict, ref_value: dict, column: str) -> pd.DataFrame:
        if not roi['Volume']:
            rel_vol = 0
        elif ref_value['Volume Units'] == 'cc':
            rel_vol = min(1, float(ref_value['Volume']) / float(roi['Volume']))
        elif ref_value['Volume Units'] == 'percent':
            rel_vol = float(ref_value['Volume']) / 100
        else:
            rel_vol = ref_value['Volume']

        if self.empty_beamset:
            dose_to_use = self.beam_set.FractionDose
            scalar = self.beam_set.FractionationPattern.NumberOfFractions
        else:
            dose_to_use = self.plan.TreatmentCourse.TotalDose
            scalar = 1

        if type(ref_value['Volume']) is str and ref_value['Volume'].casefold() == 'mean':
            planned_dose = dose_to_use.GetDoseStatistic(RoiName=roi['ROI Name'], DoseType='Average')
        else:
            planned_dose = dose_to_use.GetDoseAtRelativeVolumes(RoiName=roi['ROI Name'], RelativeVolumes=[rel_vol])[0]

        planned_dose = np.round(planned_dose * scalar)

        if column == 'Constraint':
            ref_col = 'Reference Limit'
        elif column == 'Clinical Goal':
            ref_col = 'Dose Limit'

        if column == 'Clinical Goal' and ref_value['Goal Criteria'] == 'AtLeast':
            passes = planned_dose >= int(ref_value[ref_col])
        else:
            passes = planned_dose <= int(ref_value[ref_col])

        if ref_value['Volume Units'] == 'percent':
            row['Volume'] = str(ref_value['Volume'])
        else:
            row['Volume'] = str(ref_value['Volume'])
        row['Volume Units'] = ref_value['Volume Units'].replace('percent', '%').replace('NULL', ' ')
        row[f'{column} Limit'] = int(ref_value[ref_col])
        row['Dose Planned'] = planned_dose
        if self.empty_beamset:
            row['Number Of Fx'] = self.beam_set.FractionationPattern.NumberOfFractions
        else:
            row['Number Of Fx'] = self.number_of_fx

        if column == 'Constraint':
            row['Reference'] = ref_value['Reference']
        if column == 'Clinical Goal':
            row['Reference'] = 'Clinical Goal'

        row[f'{column} Met'] = passes
        return row

    def group_dose_limits(self, constraints, clinical_goals):
        pairings = []
        paired_goals = []

        for constraint in constraints:
            paired = False
            for clinical_goal in clinical_goals:
                if constraint['Volume Units'] == clinical_goal['Volume Units']:
                    if float(constraint['Volume']) == float(clinical_goal['Volume']):
                        pairings.append((constraint, clinical_goal))
                        paired_goals.append(clinical_goal)
                        paired = True
                        break
            if not paired:
                pairings.append((constraint, None))

        for clinical_goal in clinical_goals:
            if clinical_goal not in paired_goals:
                pairings.append((None, clinical_goal))

        return pairings

    def dose_warnings(self) -> pd.DataFrame:
        rois = query_db(self.db_name, 'SELECT * FROM Rois', dicts=True)
        columns = get_columns(self.log_db_path, 'ConstraintStatus')

        df = pd.DataFrame(columns=columns)

        query = f'SELECT Fractions FROM FractionSpecification WHERE "Plan UID" = "{self.uid}"'
        override_fx = query_db(self.log_db_path, query)

        if len(override_fx):
            self.number_of_fx = int(override_fx[0][0])

        for roi in rois:
            if self.number_of_fx > 5:
                lowest_match = 'conv'
            else:
                fx_options = query_db(self.ref_db_path, 'SELECT DISTINCT("Number Of Fx") FROM DoseConstraints')
                possible_fx = [int(x[0]) for x in fx_options if x[0] != 'conv']
                lowest_match = max([x for x in possible_fx if x <= self.number_of_fx])

            name = roi['ROI Name']
            query = f'SELECT * FROM DoseConstraints WHERE "ROI Name" LIKE "{name}" AND "Number Of Fx" = "{lowest_match}"'
            constraints = query_db(self.ref_db_path, query, dicts=True)

            query = f'SELECT * FROM ClinicalGoals WHERE "ROI Name" = "{roi["ROI Name"]}"'
            clinical_goals = query_db(self.db_name, query, dicts=True)

            groupings = self.group_dose_limits(constraints, clinical_goals)

            for constraint, clinical_goal in groupings:
                row = dict.fromkeys(columns)
                row['Plan UID'] = self.uid
                row['ROI Name'] = name
                row['Clinical Goal Limit'] = ' '
                row['Clinical Goal Met'] = ' '
                row['Constraint Limit'] = ' '
                row['Constraint Met'] = ' '

                if clinical_goal:
                    row = self.check_reference_limit(row, roi, clinical_goal, column='Clinical Goal')
                if constraint:
                    row = self.check_reference_limit(row, roi, constraint, column='Constraint')

                if not clinical_goals:
                    if row['Clinical Goal Met'] is None:
                        row['Clinical Goal Met'] = 'Not Entered'
                if not row['Constraint Met']:
                    row['Status'] = 'Fail'
                elif not row['Clinical Goal Met']:
                    row['Status'] = 'Warn'
                else:
                    row['Status'] = 'Pass'

                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                df.replace({'Clinical Goal Limit': 'None'}, 'Not Entered', inplace=True)

        df['Status'] = pd.Categorical(df['Status'], ['Pass', 'Warn', 'Fail'], ordered=True)

        df.sort_values(by=['Status', 'ROI Name'], axis=0, inplace=True,
                       ignore_index=True, ascending=[False, True])

        if len(df.index):
            df = self._add_df_to_log_db('ConstraintStatus', df, return_df=True)
        return (df, lowest_match)

    def cmd_note(self) -> str:
        row = query_db(self.log_db_path, f'SELECT * FROM UserComments WHERE "Plan UID" LIKE "{self.uid}"', dicts=True)
        if len(row):
            return row[0]['CMD']
        return ''

    def md_note(self) -> str:
        row = query_db(self.log_db_path, f'SELECT * FROM UserComments WHERE "Plan UID" LIKE "{self.uid}"', dicts=True)
        if len(row):
            return row[0]['MD']
        return ''

    def roi_suggestions(self) -> pd.DataFrame:
        df = pd.DataFrame(columns=['ROI Name', 'Suggestions', 'ROI Type', 'Compliant'])

        tg263 = query_db(self.ref_db_path, 'SELECT DISTINCT("ROI Name") FROM CompliantNames')
        tg263 = [t[0].strip() for t in tg263]

        with make_conn(self.db_name, dicts=True) as conn:
            rows = conn.execute('SELECT * FROM Rois').fetchall()

        tg263_not_used = list(set(tg263) - set([x['ROI Name'] for x in rows]))

        for row in rows:
            if row['ROI Name'][:2].casefold() == 'd_':
                continue
            if row['ROI Name'].casefold() == 'couchmodel':
                continue
            if row['ROI Name'].casefold() == 'external':
                continue
            if row['ROI Type'].casefold() == 'target':
                continue
            if 'z' in row['ROI Name'][0].casefold():
                continue
            excludes = ['%', 'avoid', 'hardware', 'alara', 'bb', 'igrt', 'ptv', 'gtv',
                        'ctv', 'itv', 'prv']
            excluded = False
            for exclude in excludes:
                if exclude in row['ROI Name'].casefold():
                    excluded = True

            if excluded:
                continue

            roi = row['ROI Name'].strip()
            compliant = roi in tg263
            suggestions = []

            if not compliant:
                suggestions = difflib.get_close_matches(roi, tg263_not_used, n=5)
                if not suggestions:
                    suggestions = difflib.get_close_matches(roi, tg263_not_used, n=5, cutoff=0.33)

            row = {'ROI Name': roi,
                   'Suggestions': suggestions,
                   'ROI Type': row['ROI Type'],
                   'Compliant': compliant}
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

        df.sort_values(by=['Compliant', 'ROI Name'], ascending=[True, True], na_position='last',
                       inplace=True, ignore_index=True)

        return df

    def update_roi_names(self, roi_dict: dict) -> None:
        """Renames the patient ROIs based on a provided dictionary

        Args:
            roi_dict (dict): A renaming dictionary with format {old: new}
        """
        to_delete = []
        for original, updated in roi_dict.items():
            if updated == 'None':
                to_delete.append(original)
                continue

            try:
                query = f'UPDATE Rois SET "ROI Name" = "{updated}" WHERE "ROI Name" = "{original}"'
                replace_in_db(self.db_name, query)
                query = f'UPDATE ClinicalGoals SET "ROI Name" = "{updated}" WHERE "ROI Name" = "{original}"'
                replace_in_db(self.db_name, query)
            except Exception:
                print('Cannot write, ROIs are locked')
                to_delete.append(original)

        for roi in to_delete:
            del roi_dict[roi]

        if len(roi_dict):
            columns = get_columns(self.log_db_path, 'RoiRenaming')
            df = pd.DataFrame(columns=columns)
            for old, new in roi_dict.items():
                row = {k: v for k, v in zip(columns, [self.uid, '', old, new])}
                df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            self._add_df_to_log_db('RoiRenaming', df)

    def record_approvals(self, approvals, df) -> None:
        timestamp = isotime()
        rows = []

        for approval, data_uid in zip(approvals, df['Data UID']):
            row = {}
            row['Constraint Data UID'] = data_uid
            row['Timestamp'] = timestamp
            row['Approved'] = approval.isChecked()
            rows.append(row)
        add_rows(self.log_db_path, 'MDApprovals', rows, replace=True)

    def get_approvals(self, df) -> list:
        approvals = []
        for (_, row) in df.iterrows():
            query = f'SELECT Approved FROM MDApprovals WHERE "Constraint Data UID" = "{row["Data UID"]}"'
            results = query_db(self.log_db_path, query)
            if len(results):
                approvals.append(bool(results[0][0]))
            else:
                approvals.append(False)
        return approvals

    def record_checking(self, status_values) -> None:
        columns = get_columns(self.log_db_path, 'BeamChecking')
        row = dict.fromkeys(columns)
        row['Plan UID'] = self.uid
        row['Timestamp'] = isotime()
        for column in columns[1:]:
            if 'UID' in column or 'Timestamp' in column:
                continue
            status = status_values[column]
            row[column] = f'{status.level}; {status.text}'
        row = self._hash_row(row)
        add_rows(self.log_db_path, 'BeamChecking', row, replace=True)

    def pull_beam_checking(self, suppress_pass=True) -> dict:
        query = f'SELECT * FROM BeamChecking WHERE "Plan UID" LIKE "{self.uid}" ORDER BY Timestamp'
        all_runs = query_db(self.log_db_path, query)

        if not len(all_runs):
            return (None, None)

        latest_run = all_runs[-1]
        timestamp = latest_run[2]

        results = [Status(*x.split(';')) for x in latest_run[3:]]
        checks = get_columns(self.log_db_path, 'BeamChecking')[3:]

        passing = {}
        warning = {}
        failing = {}
        for check, result in zip(checks, results):
            if result.level == 'Pass':
                passing[check] = result
            if result.level == 'Warn':
                warning[check] = result
            if result.level == 'Fail':
                failing[check] = result
        if suppress_pass:
            output = {**failing, **warning}
        else:
            output = {**failing, **warning, **passing}
        return (timestamp, output)

    def create_ring(self, roi, inner, outer) -> None:
        requested = f'd_{roi}_Ring_{inner}->{outer}cm'
        ring_name = self.case.PatientModel.GetRoiNameWithRespectToExistingRois(DesiredRoiName=requested,
                                                                               ExaminationName=self.examination.Name)
        self.case.PatientModel.CreateRoi(Name=ring_name,
                                         Color='Green',
                                         Type='Undefined')

        ring_roi = self.case.PatientModel.StructureSets[self.examination.Name].RoiGeometries[ring_name]

        outer_expand = {'Type': 'Expand',
                        'Superior': 0,
                        'Inferior': 0,
                        'Anterior': outer,
                        'Posterior': outer,
                        'Right': outer,
                        'Left': outer
                        }

        inner_expand = {'Type': 'Expand',
                        'Superior': 0,
                        'Inferior': 0,
                        'Anterior': inner,
                        'Posterior': inner,
                        'Right': inner,
                        'Left': inner
                        }

        ring_roi.OfRoi.CreateAlgebraGeometry(Examination=self.examination,
                                             Algorithm='Auto',
                                             ExpressionA={'Operation': 'Union',
                                                          'SourceRoiNames': [roi],
                                                          'MarginSettings': outer_expand},
                                             ExpressionB={'Operation': 'Union',
                                                          'SourceRoiNames': [roi],
                                                          'MarginSettings': inner_expand},
                                             ResultOperation='Subtraction')
        return True

    def min_jaw_gap(self, progress_bar, xmin=0.5, ymin=0.5) -> Status:
        changed = False
        total_segments = 0
        for beam in self.beam_set.Beams:
            total_segments += len(beam.Segments)

        count = 0
        for beam in self.beam_set.Beams:
            for segment in beam.Segments:
                count += 1
                progress_bar.setValue(round(100 * (count + 1) / total_segments))
                x_width = segment.JawPositions[1] - segment.JawPositions[0]
                y_width = segment.JawPositions[3] - segment.JawPositions[2]

                if x_width < xmin or y_width < ymin:
                    changed = True
                    x_diff = 0
                    y_diff = 0

                    if x_width < xmin:
                        x_diff = (xmin - x_width + float(1e-6)) / 2
                    if y_width < ymin:
                        y_diff = (ymin - y_width + float(1e-6)) / 2

                    updated = [segment.JawPositions[0] - x_diff,
                               segment.JawPositions[1] + x_diff,
                               segment.JawPositions[2] - y_diff,
                               segment.JawPositions[3] + y_diff]
                    segment.JawPositions = [float(x) for x in updated]
        progress_bar.setValue(100)
        if changed:
            return Status('Warn', 'Warning: Changes Made')
        return Status('Pass', 'Pass')

    def min_mlc_gap(self, progress_bar, limit=0.05) -> Status:
        if 'Versa' in self.beam_set.MachineReference.MachineName:
            limit = 0.50

        any_changed = False

        total_segments = 0
        for beam in self.beam_set.Beams:
            total_segments += len(beam.Segments)

        count = 0
        for beam in self.beam_set.Beams:
            for segment in beam.Segments:
                count += 1
                segment_changed = False
                new_left = []
                new_right = []
                for left, right in zip(*segment.LeafPositions):
                    if abs(left - right) < limit and abs(left - right) > 0:
                        any_changed = True
                        segment_changed = True
                        mlc_diff = float(Decimal((limit - abs(left - right) + 1e-6) / 2))
                        new_left.append(left - mlc_diff)
                        new_right.append(right + mlc_diff)
                    else:
                        new_left.append(left)
                        new_right.append(right)

                if segment_changed:
                    segment.LeafPositions = [new_left, new_right]
                progress_bar.setValue(round(100 * (count + 1) / total_segments))

        if any_changed:
            return Status('Warn', 'Warning: Changes Made')
        progress_bar.setValue(100)
        return Status('Pass', 'Pass')

    def max_mlc_span(self, progress_bar, limit=15) -> Status:
        total_beams = len(self.beam_set.Beams)
        for i, beam in enumerate(self.beam_set.Beams):
            progress_bar.setValue(round(100 * (i + 1) / total_beams))
            for segment in beam.Segments:
                bankA = segment.LeafPositions[0]
                bankB = segment.LeafPositions[1]

                bankA_min = np.ceil(max(bankA) * 100) / 100
                bankA_max = np.floor(max(bankB) * 100) / 100

                bankB_min = np.ceil(max(bankB) * 100) / 100
                bankB_max = np.floor(max(bankB) * 100) / 100

                bankA_span = bankA_max - bankA_min
                bankB_span = bankB_max - bankB_min

                if bankA_span > limit or bankB_span > limit:
                    # Fix span limit?
                    pass

    def mlc_behind_jaws(self, progress_bar, tolerance=0.10, gap=0.60) -> Status:
        total_segments = 0
        for beam in self.beam_set.Beams:
            total_segments += len(beam.Segments)

        outside = 0
        count = 0
        issue_leafs = []
        for beam in self.beam_set.Beams:
            for i, segment in enumerate(beam.Segments):
                count += 1
                jaw_x0, jaw_x1 = segment.JawPositions[:2]
                for j, (mlc_l, mlc_r) in enumerate(zip(*segment.LeafPositions)):
                    if abs(mlc_r - mlc_l) < gap:
                        continue
                    left_margin = jaw_x0 - tolerance
                    right_margin = jaw_x1 + tolerance
                    if mlc_l <= (left_margin - gap) and mlc_r <= left_margin:
                        outside += 1
                        issue_leafs.append(IssueLeaf(beam.Name, i+1, j+1, 'A'))
                    elif mlc_l >= right_margin and mlc_r >= (right_margin + gap):
                        outside += 1
                        issue_leafs.append(IssueLeaf(beam.Name, i+1, j+1, 'B'))
                progress_bar.setValue(round(100 * (count + 1) / total_segments))
        progress_bar.setValue(100)
        if outside:
            issue_leafs.sort()
            issue_beams = ','.join(list(set([x.beam_name for x in issue_leafs])))
            noun = 'Leaf' if outside == 1 else 'Leaves'
            return issue_leafs, Status('Warn', f'{noun} past jaws on {issue_beams}')
        return None, Status('Pass', 'Pass')

    def collimator_check(self, progress_bar) -> Status:
        if self.beam_set.DeliveryTechnique in ['DynamicArc', 'StaticArc']:
            used_angles = []
            for beam in self.beam_set.Beams:
                collimator = beam.InitialCollimatorAngle
                if collimator == 0:
                    progress_bar.setValue(100)
                    return Status('Fail', 'Fail: Zero collimator angles on a beam')
                if collimator in used_angles:
                    progress_bar.setValue(100)
                    return Status('Warn', 'Warning: Same collimator angles on some beams')
                used_angles.append(collimator)
        progress_bar.setValue(100)
        return Status('Pass', 'Pass')

    def fif_check(self, progress_bar) -> Status:
        if self.beam_set.DeliveryTechnique != 'SMLC':
            progress_bar.setValue(100)
            return Status('Pass', 'Pass')

        total_beams = len(self.beam_set.Beams)
        for i, beam in enumerate(self.beam_set.Beams):
            progress_bar.setValue(round(100 * (i + 1) / total_beams))
            if beam.BeamMU < 1:
                return Status('Fail', 'FiF MU < 1')
            for segment in beam.Segments:
                segment_MU = segment.RelativeWeight * beam.BeamMU
                if segment_MU < 1:
                    progress_bar.setValue(100)
                    return Status('Fail', 'Fail: FiF MU < 1')
        progress_bar.setValue(100)
        return Status('Pass', 'Pass')

    def algorithm_check(self, progress_bar) -> Status:
        progress_bar.setValue(100)
        if self.empty_beamset and not self.beam_set.FractionDose.DoseValues.IsClinical:
            return Status('Fail', 'Final clinical dose not computed.')
        elif not self.empty_beamset and not self.plan.TreatmentCourse.TotalDose.DoseValues.IsClinical:
            return Status('Fail', 'Final clinical dose not computed.')
        return Status('Pass', 'Pass')

    def edw_check(self, progress_bar) -> Status:
        # Check that the field MU > 20 for EDW
        # Set progress bar to 100 if it is skipped
        total_beams = len(self.beam_set.Beams)
        for i, beam in enumerate(self.beam_set.Beams):
            progress_bar.setValue(round(100 * (i + 1) / total_beams))

            if beam.Wedge is None:
                continue
            elif 'Varian' in beam.Wedge.Type:
                y1 = beam.Segments[0].JawPositions[2]
                y2 = beam.Segments[0].JawPositions[3]
                y_opening = abs(y2 - y1)

                if beam.BeamMU < 20:
                    return Status('Fail', 'Wedged field MU < 20')
                if y_opening > 30 or y_opening < 4:
                    return Status('Fail', 'Y Opening <4cm or >30cm')
                if beam.Wedge.Orientation == 'In':
                    if y1 < -10 or y1 > 10:
                        return Status('Fail', 'Y1 not between -10 and 10 cm')
                if beam.Wedge.Orientation == 'Out':
                    if y2 < -10 or y2 > 10:
                        return Status('Fail', 'Y2 not between -10 and 10 cm')
            elif 'Elketa' in beam.Wedge.Type:
                if beam.BeamMU > 999:
                    return Status('Fail', 'Wedged field MU > 999')
        return Status('Pass', 'Pass')

    def cw_check(self, progress_bar) -> Status:
        progress_bar.setValue(100)
        if self.beam_set.DeliveryTechnique != 'DynamicArc' and self.beam_set.DeliveryTechnique != 'ConformalArc':
            return Status('Pass', 'Pass')

        previous = None
        same_rotation = False
        for beam in self.beam_set.Beams:
            direction = beam.ArcRotationDirection
            if direction == previous:
                same_rotation = True
            previous = direction
            if direction == 'Clockwise' and 'CCW' in beam.Name:
                return Status('Warn', 'Clockwise beam named CCW')
            if direction == 'CounterClockwise' and 'CW' in beam.Name and 'CCW' not in beam.Name:
                return Status('Warn', 'CounterClockwise beam named CW')
        if same_rotation:
            return Status('Warn', 'Warning: CW or CCW back-to-back')
        return Status('Pass', 'Pass')

    def rx_check(self, progress_bar) -> Status:
        if self.beam_set.FractionationPattern is None:
            progress_bar.setValue(100)
            return Status('Fail', 'No fractionation entered')

        fractions = self.beam_set.FractionationPattern.NumberOfFractions
        total_dose = self.beam_set.Prescription.PrimaryPrescriptionDoseReference.DoseValue
        dose_per_fx = total_dose / fractions

        progress_bar.setValue(100)
        if dose_per_fx < 150:
            return Status('Warn', 'Dose per fx < 150')
        return Status('Pass', 'Pass')

    def _box_contains(self, lower, upper, dmax_coords):
        if lower['x'] <= dmax_coords['x'] <= upper['x']:
            if lower['y'] <= dmax_coords['y'] <= upper['y']:
                if lower['z'] <= dmax_coords['z'] <= upper['z']:
                    return True
        return False

    def dmax_check(self, progress_bar) -> Status:
        if self.empty_beamset:
            total_dose = self.beam_set.FractionDose
        else:
            total_dose = self.plan.TreatmentCourse.TotalDose
        dmax_coords = total_dose.GetCoordinateOfMaxDose()
        frame = total_dose.InDoseGrid.FrameOfReference
        dmax = total_dose.InterpolateDoseInPoint(Point=dmax_coords,
                                                 PointFrameOfReference=frame)
        struct_sets = self.case.PatientModel.StructureSets
        # Determine which CT was used for dose calculation and only use that set

        # Pull the PTV or volume used for prescription
        within_organ = []
        within_target = []
        total_rois = 0
        for structs in struct_sets:
            total_rois += len(structs.RoiGeometries)

        counter = 0
        for structs in struct_sets:
            for roi in structs.RoiGeometries:
                counter += 1
                progress_bar.setValue(round(100 * counter / total_rois))

                name = roi.OfRoi.Name
                if not roi.HasContours():
                    continue
                if roi.OfRoi.Type not in ['Organ', 'Ptv', 'Ctv', 'Gtv']:
                    continue
                if 'd_' == name[:2].lower():
                    continue

                lower, upper = roi.GetBoundingBox()
                if self._box_contains(lower, upper, dmax_coords):
                    dose = total_dose.GetDoseStatistic(RoiName=name,
                                                       DoseType='Max')
                    if dose != dmax:
                        continue
                    if roi.OfRoi.Type == 'Organ':
                        within_organ.append(name)
                    else:
                        within_target.append(name)

        if len(within_organ):
            within_organ.sort()
            roi_fmt = ','.join(within_organ)
            return Status('Warn', f'Dmax within Organs: {roi_fmt}')

        if not len(within_target):
            within_organ.sort()
            roi_fmt = ','.join(within_target)
            return Status('Warn', f'Dmax not within "Targets" type structure')
        return Status('Pass', 'Pass')

    def dose_grid_size(self, progress_bar) -> Status:
        if self.empty_beamset:
            voxel_size = self.beam_set.FractionDose.InDoseGrid.VoxelSize
        else:
            voxel_size = self.plan.TreatmentCourse.TotalDose.InDoseGrid.VoxelSize
        dx = voxel_size['x']
        dy = voxel_size['y']
        dz = voxel_size['z']
        progress_bar.setValue(100)
        if dx > 0.3 or dy > 0.3 or dz > 0.3:
            return Status('Fail', 'At least one dose grid dimension is >3mm')
        if dx == 0.3 or dy == 0.3 or dz == 0.3:
            if self.number_of_fx > 5:
                return Status('Warn', 'At least one dose grid dimension is 3mm')
            return Status('Fail', 'A 3mm dose grid is not appropriate for SBRT')
        return Status('Pass', 'Pass')

    def collision_check(self, progress_bar):
        isocenter_loc = self.beam_set.Beams[0].Isocenter.Position
        isocenter_name = self.beam_set.Beams[0].Isocenter.Annotation.Name
        ring_desired = f'd_CollisionCheckRing_{isocenter_name}'
        couch_desired = f'd_PossibleCouchCollision_{isocenter_name}'
        external_desired = f'd_PossibleExternalCollision_{isocenter_name}'

        if self._approved():
            ring = self.case.PatientModel.GetRoiNameWithRespectToExistingRois(DesiredRoiName=ring_desired,
                                                                              ExaminationName=self.examination.Name)
            couch = self.case.PatientModel.GetRoiNameWithRespectToExistingRois(DesiredRoiName=couch_desired,
                                                                               ExaminationName=self.examination.Name)
            external = self.case.PatientModel.GetRoiNameWithRespectToExistingRois(DesiredRoiName=external_desired,
                                                                                  ExaminationName=self.examination.Name)
        else:
            ring = ring_desired
            couch = couch_desired
            external = external_desired

        all_names = [x.Name for x in self.case.PatientModel.RegionsOfInterest]

        exam_name = self.examination.Name

        for name in [ring, couch, external]:
            if name in all_names:
                self.case.PatientModel.StructureSets[exam_name].RoiGeometries[name].OfRoi.DeleteRoi()

            self.case.PatientModel.CreateRoi(Name=name,
                                             Color='Red',
                                             Type='Undefined')

        roi_ring = self.case.PatientModel.StructureSets[exam_name].RoiGeometries[ring]
        roi_couch = self.case.PatientModel.StructureSets[exam_name].RoiGeometries[couch]
        roi_external = self.case.PatientModel.StructureSets[exam_name].RoiGeometries[external]

        # Make the axis align with the couch angle
        couch_rotations = []
        for beam in self.beam_set.Beams:
            couch_rotations.append(beam.CouchRotationAngle)

        if set(couch_rotations) - set([0, 90, 270]):
            return Status('Warn', 'Couch angles not supported. Check for collisions')

        # I can also handle this as translating the ROI by the given couch rotation value
        couch_rotations = list(set(couch_rotations))

        for i, angle in enumerate(couch_rotations):
            progress_bar.setValue(round(100 * i / len(couch_rotations)))

            if angle == 0:
                axis = {'x': 0, 'y': 0, 'z': 1}
            elif angle == 90 or angle == 270:
                axis = {'x': 1, 'y': 0, 'z': 0}

            # VersaC Couch Height = 42.5 cm, Angle = +/- 45 degrees
            # TrueBeam Couch Height = 39.4 cm, Angle = +/- 00 degrees
            # Need to check within whatever degree range, not passing through 180

            scan_length = self.examination.Series[0].ImageStack.SlicePositions[-1]
            roi_ring.OfRoi.CreateCylinderGeometry(Radius=39.5,
                                                  Axis=axis,
                                                  Length=2 * int(scan_length),
                                                  Center=isocenter_loc,
                                                  Examination=self.examination)

            self.case.PatientModel.StructureSets[exam_name].SimplifyContours(RoiNames=[ring],
                                                                             MaxNumberOfPoints=10000,
                                                                             ReduceMaxNumberOfPointsInContours=True)

            self.case.PatientModel.StructureSets[exam_name].SimplifyContours(RoiNames=['CouchModel'],
                                                                             MaxNumberOfPoints=100000,
                                                                             ReduceMaxNumberOfPointsInContours=True)

            no_expansion = {'Type': 'Expand',
                            'Superior': 0,
                            'Inferior': 0,
                            'Anterior': 0,
                            'Posterior': 0,
                            'Right': 0,
                            'Left': 0
                            }

            roi_couch.OfRoi.CreateAlgebraGeometry(Examination=self.examination,
                                                  Algorithm='Auto',
                                                  ExpressionA={'Operation': 'Union',
                                                               'SourceRoiNames': ['CouchModel'],
                                                               'MarginSettings': no_expansion},
                                                  ExpressionB={'Operation': 'Union',
                                                               'SourceRoiNames': [ring],
                                                               'MarginSettings': no_expansion},
                                                  ResultOperation='Subtraction')
            roi_external.OfRoi.CreateAlgebraGeometry(Examination=self.examination,
                                                     Algorithm='Auto',
                                                     ExpressionA={'Operation': 'Union',
                                                                  'SourceRoiNames': ['External'],
                                                                  'MarginSettings': no_expansion},
                                                     ExpressionB={'Operation': 'Union',
                                                                  'SourceRoiNames': [ring],
                                                                  'MarginSettings': no_expansion},
                                                     ResultOperation='Subtraction')
            collision_possible = []
            if roi_external.HasContours():
                collision_possible.append('External')
            else:
                roi_external.OfRoi.DeleteRoi()

            if roi_couch.HasContours():
                collision_possible.append('Couch')
            else:
                roi_couch.OfRoi.DeleteRoi()

            if len(collision_possible):
                total_dose = self.plan.PlanOptimizations[0].TreatmentCourseSource.TotalDose
                total_dose.UpdateDoseGridStructures()
                progress_bar.setValue(100)
                return Status('Warn', f'Collision possible with {",".join(collision_possible)}. Check arcs.')

        self.update_roi_dose_statistics()

        progress_bar.setValue(100)
        return Status('Pass', 'Pass')

    def update_roi_dose_statistics(self):
        try:
            total_dose = self.plan.PlanOptimizations[0].TreatmentCourseSource.TotalDose
            total_dose.UpdateDoseGridStructures()
        except Exception:
            pass

    def partial_arc_check(self, progress_bar) -> Status:
        progress_bar.setValue(0)
        isocenter = self.beam_set.Beams[0].Isocenter.Position
        ct_origin = None
        exam_name = self.examination.Name

        if self.beam_set.DeliveryTechnique != 'DynamicArc' or self.beam_set.DeliveryTechnique != 'ConformalArc':
            progress_bar.setValue(100)
            return Status('Pass', 'Pass')

        for poi in self.case.PatientModel.StructureSets[exam_name].PoiGeometries:
            if poi.OfPoi.Type == 'LocalizationPoint':
                ct_origin = poi.Point

        if ct_origin is None:
            progress_bar.setValue(100)
            return Status('Warn', 'No POI of type Localization Point')

        lat_diff = abs(ct_origin['x'] - isocenter['x'])

        total_segments = 0
        for beam in self.beam_set.Beams:
            total_segments += len(beam.Segments)

        if lat_diff > 5.5:
            if self.examination.PatientPosition == 'HFS':
                to_left = isocenter['x'] > ct_origin['x']
            elif self.examination.PatientPosition == 'FFS':
                to_left = isocenter['x'] < ct_origin['x']
            else:
                progress_bar.setValue(100)
                return Status('Warn', 'Only HFS / FFS supported in this test')
            segment_count = 0
            for beam in self.beam_set.Beams:
                initial_gantry = beam.GantryAngle
                current_gantry = initial_gantry
                for segment in beam.Segments:
                    segment_count += 1
                    progress_bar.setValue(round(100 * segment_count / total_segments))
                    current_gantry += segment.DeltaGantryAngle
                    left_fail = to_left and 85 < current_gantry < 95
                    right_fail = not to_left and 265 < current_gantry < 275
                    if left_fail or right_fail:
                        progress_bar.setValue(100)
                        return Status('Warn', 'Gantry arc contralateral to shift > 5cm')
        progress_bar.setValue(100)
        return Status('Pass', 'Pass')

    def vertical_clearance(self, progress_bar) -> Status:
        progress_bar.setValue(0)
        isocenter = self.plan.BeamSets[0].Beams[0].Isocenter.Position
        exam_name = self.examination.Name
        roi_couch = self.case.PatientModel.StructureSets[exam_name].RoiGeometries['CouchModel']
        _, upper = roi_couch.GetBoundingBox()
        separation = abs(abs(isocenter['y']) - abs(upper['y']))

        progress_bar.setValue(100)
        if separation > 24:
            return Status('Warn', 'Isocenter >24cm above couch top.')
        return Status('Pass', 'Pass')

    @property
    def empty_beamset(self) -> bool:
        for beamset in self.plan.BeamSets:
            print(beamset.DicomPlanLabel, len(beamset.Beams))
            if not len(beamset.Beams):
                return True
        return False

    def recompute_dose(self, progress_bar) -> Status:
        progress_bar.setValue(0)

        if self.beam_set.FractionDose.DoseValues is None:
            if self.beam_set.Modality == 'Electrons':
                self.beam_set.ComputeDose(ComputeBeamDoses=True,
                                          DoseAlgorithm='ElectronMonteCarlo',
                                          ForceRecompute=False)
            else:
                self.beam_set.ComputeDose(ComputeBeamDoses=True,
                                          DoseAlgorithm='CCDose',
                                          ForceRecompute=False)
            self._pull_clinical_goals()
            progress_bar.setValue(100)
            return Status('Warn', 'Dose recomputed - Please Check!')
        progress_bar.setValue(100)
        if self.empty_beamset:
            return Status('Warn', 'Empty beamset. Dose stats ONLY on this beamset.')
        return Status('Pass', 'No recomputation needed')

    def merge_bilateral_structures(self):
        rois = [x[0] for x in query_db(self.db_name, 'SELECT * FROM Rois')]

        pairings = {}
        for roi in rois:
            if '_l' in roi.lower() or '_r' in roi.lower():
                base = roi.rsplit('_', 1)[0]
                if base.lower() not in ['femur', 'kidney', 'lung']:
                    continue
                if base in pairings:
                    pairings[base].append(roi)
                else:
                    pairings[base] = [roi]

        for combined, references in pairings.items():
            if len(references) < 2:
                continue

            new_name = combined + 's'

            if new_name in rois:
                continue

            colors = ['Red', 'Blue', 'Green', 'Yellow', 'Orange']
            color = random.choice(colors)

            self.case.PatientModel.CreateRoi(Name=new_name,
                                             Color=color,
                                             Type='Organ')

            no_expansion = {'Type': 'Expand',
                            'Superior': 0,
                            'Inferior': 0,
                            'Anterior': 0,
                            'Posterior': 0,
                            'Right': 0,
                            'Left': 0
                            }

            exam_name = self.examination.Name
            new_roi = self.case.PatientModel.StructureSets[exam_name].RoiGeometries[new_name]

            new_roi.OfRoi.CreateAlgebraGeometry(Examination=self.examination,
                                                Algorithm='Auto',
                                                ExpressionA={'Operation': 'Union',
                                                             'SourceRoiNames': [*references[1:]],
                                                             'MarginSettings': no_expansion},
                                                ExpressionB={'Operation': 'Union',
                                                             'SourceRoiNames': [references[0]],
                                                             'MarginSettings': no_expansion},
                                                ResultOperation='Union')

    def bolus_check(self, progress_bar) -> Status:
        has_bolus = []
        for i, beam in enumerate(self.beam_set.Beams):
            progress_bar.setValue((i+1)/len(self.beam_set.Beams))
            if beam.Boli:
                has_bolus.append(True)
            else:
                has_bolus.append(False)

        progress_bar.setValue(100)

        if sum(has_bolus) and sum(has_bolus) != len(has_bolus):
            return Status('Warn', 'Not all beams have bolus')
        return Status('Pass', 'Pass')

    def categorize_targets(self):
        target_types = ['Gtv', 'Ptv', 'Ctv', 'Itv', 'Tumor', 'SurgicalBed']
        rois = [x[0] for x in query_db(self.db_name, 'SELECT * FROM Rois')]
        oars = [x[0] for x in query_db(self.ref_db_path, 'SELECT "ROI Name" FROM DoseConstraints')]

        for roi in rois:
            if 'd_' in roi.casefold():
                continue
            if 'zz' in roi.casefold()[:2]:
                continue
            if 'LN_Neck' in roi.casefold():
                continue
            if roi in oars:
                self.case.PatientModel.RegionsOfInterest[roi].Type = 'Organ'
                self.case.PatientModel.RegionsOfInterest[roi].OrganData.OrganType = 'OrganAtRisk'
            else:
                for target_type in target_types:
                    if target_type.casefold() in roi.casefold():
                        self.case.PatientModel.RegionsOfInterest[roi].OrganData.OrganType = 'Target'
                        if target_type in ['Itv', 'Tumor', 'SurgicalBed']:
                            self.case.PatientModel.RegionsOfInterest[roi].Type = 'TreatedVolume'
                        else:
                            self.case.PatientModel.RegionsOfInterest[roi].Type = target_type

    def _get_current_letter(self, start, index):
        if start is None:
            start_index = 1
            repeats = 1
        else:
            start_index = ord(start[0].upper()) - 64 + 1
            repeats = start_index // 26 + len(start)
        current = chr((start_index + index) % 26 + 64) * repeats
        return current

    def create_setup_beams(self, start=None):
        if self.beam_set.PatientPosition == 'HeadFirstSupine':
            angles = [0, 90, 270, 180, 0]
        elif self.beam_set.PatientPosition == 'HeadFirstProne':
            angles = [180, 270, 90, 0, 0]
        elif self.beam_set.PatientPosition == 'FeetFirstSupine':
            angles = [0, 270, 90, 180, 0]
        elif self.beam_set.PatientPosition == 'FeetFirstProne':
            angles = [180, 90, 270, 0, 0]
        else:
            return False

        self.beam_set.UpdateSetupBeams(ResetSetupBeams=True, SetupBeamsGantryAngles=angles)
        names = ['AP', 'LLAT', 'RLAT', 'PA', 'CT']
        setup_beams = self.beam_set.PatientSetup.SetupBeams
        for i, beam in enumerate(setup_beams):
            prefix = self._get_current_letter(start, i)
            name = names[i % len(names)]
            beam.Name = f'{prefix}_{name}'
            beam.Description = name
            beam.Segments[0].JawPositions = [-10, 10, -10, 10]
        return True

    def set_color_table(self):
        color_table = self.case.CaseSettings.DoseColorMap.ColorTable
        color_table.clear()
        self.case.CaseSettings.DoseColorMap.PresentationType = 'Relative'

        if self.case.Physician.Name and len(self.case.Physician.Name):
            physician = self.case.Physician.Name.casefold()
        else:
            physician = ''

        head_neck_attendings = [None]
        if any([lastname in physician for lastname in head_neck_attendings]):
            dose_rgb = {110.062893081761: [128, 0, 128],
                        105.774728416238: [128, 255, 128],
                        100: [255, 0, 0],
                        94.3396226415094: [128, 255, 255],
                        89.622641509434: [0, 128, 255],
                        84.9056603773585: [0, 0, 255],
                        77.3584905660377: [255, 255, 0],
                        64.3224699828473: [255, 128, 0],
                        50.0285877644368: [128, 64, 0],
                        28.587764436821: [255, 255, 255]}

            for dose, rgb in dose_rgb.items():
                color_table[dose] = System.Drawing.Color.FromArgb(255, *rgb)
        else:
            color_dose = {'Purple': 115,
                          'DarkViolet': 107,
                          'Red': 100,
                          'Orange': 95,
                          'Yellow': 90,
                          'YellowGreen': 85,
                          'Green': 80,
                          'SkyBlue': 70,
                          'Blue': 50,
                          'DarkBlue': 30}

            breast_attendings = [None]
            if any([lastname in physician for lastname in breast_attendings]):
                color_dose['Purple'] = 107
                color_dose['DarkViolet'] = 105

            for color, dose in color_dose.items():
                color_table[dose] = color

        self.case.CaseSettings.DoseColorMap.ColorTable = color_table
        self.case.CaseSettings.DoseColorMap.PresentationType = 'Absolute'
        if self.plan is None or not self.plan.TreatmentCourse.TotalDose.DoseValues:
            self.case.CaseSettings.DoseColorMap.ColorMapReferenceType = 'RelativePrescription'

    def _drr_list(self, beams, drr_setting):
        drrs = []
        for beam in beams:
            if 'CT' in beam.Name:
                continue
            drrs.append(f'{self.plan.Name}:{self.beam_set.DicomPlanLabel}:{beam.Name}:{drr_setting}')
        return drrs

    def drr_options(self):
        return [x.Name for x in self.beam_set.DrrSettings]

    def alias_options(self):
        return [self.beam_set.MachineReference.MachineName]

    def alias_value(self):
        if hasattr(self.beam_set.DicomExportProperties, 'ExportedTreatmentMachineName'):
            return self.beam_set.DicomExportProperties.ExportedTreatmentMachineName
        return None

    def igrt_probable(self):
        query = 'SELECT * FROM IGRTStructures'
        results = query_db(self.ref_db_path, query, dicts=False)
        return [x[0] for x in results]

    def set_roi_poi_visibility(self, rois, pois):
        with CompositeAction('Toggle Visiblity'):
            for roi in self.case.PatientModel.RegionsOfInterest:
                if rois is not None and roi.Name in rois:
                    self.patient.SetRoiVisibility(RoiName=roi.Name, IsVisible=True)
                else:
                    self.patient.SetRoiVisibility(RoiName=roi.Name, IsVisible=False)

            for poi in self.case.PatientModel.PointsOfInterest:
                if pois is not None and poi.Name in pois:
                    self.patient.SetPoiVisibility(PoiName=poi.Name, IsVisible=True)
                else:
                    self.patient.SetPoiVisibility(PoiName=poi.Name, IsVisible=False)

    def visible_pois(self):
        visibles = {}
        for poi in self.case.PatientModel.PointsOfInterest:
            visibles[poi] = self.patient.GetPoiVisibility(PoiName=poi.Name)

    def visible_rois(self):
        visibles = {}
        for roi in self.case.PatientModel.RegionsOfInterest:
            visibles[roi] = self.patient.GetRoiVisibility(RoiName=roi.Name)

    def all_rois(self):
        all_rois = []
        for roi in self.case.PatientModel.RegionsOfInterest:
            all_rois.append(roi.Name)
        return all_rois

    def igrt_eligible(self):
        igrt = []
        for roi in self.case.PatientModel.RegionsOfInterest:
            if roi.Type not in ['Target', 'Ptv', 'Ctv', 'Gtv']:
                igrt.append(roi.Name)
        igrt.sort()
        return igrt

    def targets_eligible(self):
        targets = []
        for roi in self.case.PatientModel.RegionsOfInterest:
            if roi.OrganData.OrganType == 'Target':
                targets.append(roi.Name)
        targets.sort()
        return targets

    def versa_roi_warning(self, targets_to_exclude, igrt_selected):
        if 'Versa' not in self.beam_set.MachineReference.MachineName:
            return False
        total_rois = 0
        for roi in self.case.PatientModel.RegionsOfInterest:
            if roi.OrganData.OrganType in ['Target', 'Ptv', 'Ctv', 'Gtv']:
                if targets_to_exclude is None or roi.Name not in targets_to_exclude:
                    total_rois += 1
            if igrt_selected is not None and roi.Name in igrt_selected:
                total_rois += 1
        return total_rois >= 10

    def exclude_rois_pois(self, igrt_structures, targets_to_exclude=None,
                          drr_structures=None, include_targets=True,
                          pois_to_exclude=None) -> None:
        with CompositeAction('Toggle Exports'):
            excluded_rois = []
            excluded_pois = pois_to_exclude
            for roi in self.case.PatientModel.RegionsOfInterest:
                if roi.OrganData.OrganType in ['Target', 'Ptv', 'Ctv', 'Gtv']:
                    if not include_targets:
                        excluded_rois.append(roi.Name)
                    elif targets_to_exclude is not None and roi.Name in targets_to_exclude:
                        excluded_rois.append(roi.Name)
                    continue
                if roi.RoiMaterial is not None:  # Cannot exclude materials with override
                    continue
                if igrt_structures is None or roi.Name not in igrt_structures:
                    excluded_rois.append(roi.Name)

            if len(excluded_rois) or len(excluded_pois):
                self.case.PatientModel.ToggleExcludeFromExport(ExcludeFromExport=True,
                                                               RegionOfInterests=excluded_rois,
                                                               PointsOfInterests=excluded_pois)

    def include_all_rois_pois(self):
        with CompositeAction('Toggle Exports'):
            rois = [roi.Name for roi in self.case.PatientModel.RegionsOfInterest]
            pois = [poi.Name for poi in self.case.PatientModel.PointsOfInterest]
            self.case.PatientModel.ToggleExcludeFromExport(ExcludeFromExport=False,
                                                           RegionOfInterests=rois,
                                                           PointsOfInterests=pois)

    def export(self, location='', mv_ports=False, drr_setting='', igrt_structures=None,
               targets_to_exclude=None, pois_to_exclude=None, machine_alias=None) -> None:
        export_parameters = {}
        export_parameters['Examinations'] = [self.examination.Name]
        export_parameters['RtStructureSetsForExaminations'] = [self.examination.Name]
        export_parameters['BeamSets'] = [f'{self.plan.Name}:{self.beam_set.DicomPlanLabel}']
        export_parameters['IgnorePreConditionWarnings'] = True

        self.include_all_rois_pois()

        if location == 'MIM':
            export_parameters['Connection'] = {'Node': '',
                                               'Port': 0,
                                               'CallingAE': '',
                                               'CalledAE': '',
                                               'Title': ''}
            export_parameters['PhysicalBeamSetDoseForBeamSets'] = [f'{self.plan.Name}:{self.beam_set.DicomPlanLabel}']
        elif location == 'ARIA':
            export_parameters['Connection'] = {'Node': '',
                                               'Port': 0,
                                               'CallingAE': '',
                                               'CalledAE': '',
                                               'Title': ''}

            # self.case.PatientModel.RoiVisualizationSettings(...)
            self.exclude_rois_pois(igrt_structures=igrt_structures, targets_to_exclude=targets_to_exclude,
                                   pois_to_exclude=pois_to_exclude)

            if mv_ports:
                export_parameters['TreatmentBeamDrrImages'] = self._drr_list(self.beam_set.Beams, drr_setting)
            export_parameters['SetupBeamDrrImages'] = self._drr_list(self.beam_set.PatientSetup.SetupBeams, drr_setting)

            self.beam_set.SetDicomExportProperties(UseStereotacticApplicatorTypeForPhotonCones=False,
                                                   ExportedTreatmentMachineName=machine_alias)
            self.beam_set.SetRayCareCustomLabels(PlanLabels=[], DoseLabels=[], DrrLabels=[])
        elif location == 'Mobius':
            export_parameters['Connection'] = {'Node': '',
                                               'Port': 0,
                                               'CallingAE': '',
                                               'CalledAE': '',
                                               'Title': ''}
            export_parameters['PhysicalBeamSetDoseForBeamSets'] = [f'{self.plan.Name}:{self.beam_set.DicomPlanLabel}']
            export_parameters['PhysicalBeamDosesForBeamSets'] = [f'{self.plan.Name}:{self.beam_set.DicomPlanLabel}']
        elif location == 'AlignRT':
            export_parameters['Examinations'] = []
            export_parameters['Connection'] = {'Node': '',
                                               'Port': 0,
                                               'CallingAE': '',
                                               'CalledAE': '',
                                               'Title': ''}
            self.exclude_rois_pois(igrt_structures=igrt_structures, include_targets=False, pois_to_exclude=[])
        try:
            self.patient.Save()
            self.case.ScriptableDicomExport(**export_parameters)
        finally:
            self.include_all_rois_pois()

    def create_qa_plan(self):
        n_qa_plans = self.plan.VerificationPlans.Count
        plan_name = f'QA_{n_qa_plans + 1}'

        self.beam_set.CreateQAPlan(PhantomName='',
                                   PhantomId='',
                                   QAPlanName=plan_name,
                                   IsoCenter={'x': -0.05, 'y': -12.3, 'z': -0.19},
                                   DoseGrid={'x': 0.2, 'y': 0.2, 'z': 0.2},
                                   ComputeDoseWhenPlanIsCreated=True)
