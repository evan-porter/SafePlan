import sys
import numpy as np
import connect
from backend import BackendManager
from PyQt6.QtCore import Qt
from PyQt6 import QtCore
from PyQt6 import QtGui
from PyQt6 import QtWidgets


ALIGN_LEFT = QtCore.Qt.AlignmentFlag.AlignLeft
ALIGN_RIGHT = QtCore.Qt.AlignmentFlag.AlignRight
ALIGN_CENTER = QtCore.Qt.AlignmentFlag.AlignCenter
STATUS_COLORS = {'Pass': QtGui.QColor('#8aff96'),
                 'Warn': QtGui.QColor('#f7f692'),
                 'Fail': QtGui.QColor('#f59d9d'),
                 'Yes': QtGui.QColor('#8aff96'),
                 'No': QtGui.QColor('#f59d9d')}


sys._excepthook = sys.excepthook


def exception_hook(exctype, value, traceback):
    fmt_message = f'{exctype} {value} {traceback}'
    print(fmt_message)
    sys._excepthook(exctype, value, traceback)
    sys.exit(1)


sys.excepthook = exception_hook


class RestrictedScrollQComboBox(QtWidgets.QComboBox):
    def __init__(self, scrollWidget=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scrollWidget = scrollWidget
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)

    def wheelEvent(self, *args, **kwargs):
        if self.hasFocus():
            return QtWidgets.QComboBox.wheelEvent(self, *args, **kwargs)
        else:
            return self.scrollWidget.wheelEvent(*args, **kwargs)


class ResizableTableView(QtWidgets.QTableView):
    def __init__(self):
        super().__init__()

    def sizeHint(self, xpad=0, ypad=0):
        h = self.horizontalHeader()
        v = self.verticalHeader()
        f = self.frameWidth() * 2
        total_width = min(900, h.length() + v.sizeHint().width() + f + 50)
        total_height = min(200, v.length() + h.sizeHint().height() + f + 25)
        return QtCore.QSize(total_width + xpad, total_height + ypad)


class BasicTable(QtCore.QAbstractTableModel):
    def __init__(self, data, allow_approvals=False):
        super().__init__()
        self._data = data
        self.allow_approvals = allow_approvals
        self.approvals = []

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])

    def rowCount(self, index):
        return self._data.shape[0]

    def columnCount(self, index):
        return self._data.shape[1]

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._data.columns[section])

            if orientation == Qt.Orientation.Vertical:
                return str(self._data.index[section])


class ColorTable(BasicTable):
    def __init__(self, data, allow_approval=False):
        super().__init__(data, allow_approval)
        self._data = data

    def data(self, index, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if not index.column() and self.allow_approvals:
                cbox = QtWidgets.QCheckBox()
                self.approvals.append(cbox)
                return cbox
            return str(self._data.iloc[index.row(), index.column()])

        if role == Qt.ItemDataRole.BackgroundRole:
            if 'Status' in self._data:
                status = self._data.iloc[index.row()]['Status']
            else:
                status = self._data.iloc[index.row()]['Compliant']
            return STATUS_COLORS[status]


class ROIRenaming(QtWidgets.QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.backend = backend
        self.instructions = 'ROIs with suggestions may need renaming.\nSelecting "Keep" leaves the ROI unchanged'
        self._acknowledged = False
        self._user_selected = {}

    def _approval_check(self):
        label = QtWidgets.QLabel('Please un-approve the contours before running')
        self.layout.addWidget(label, 1, 0, 1, 2)

        button = QtWidgets.QPushButton('Close', self)
        button.clicked.connect(self.close)
        button.setFixedSize(100, 30)
        self.layout.addWidget(button, 2, 0, 1, 2, alignment=ALIGN_CENTER)

    def _proceed(self):
        self._acknowledged = True
        self.layout.removeWidget(self.label)
        self.label.deleteLater()

        self.layout.removeWidget(self.checkbox)
        self.checkbox.deleteLater()

        self.layout.removeWidget(self.button)
        self.button.deleteLater()

        self._renaming_table()
        self.resize(self._sizeHint())

    def _state_changed(self):
        self.layout.removeWidget(self.button)
        self.button.deleteLater()

        self.button = QtWidgets.QPushButton('Proceed', self)
        self.button.clicked.connect(self._proceed)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

    def _experimental_check(self):
        line1 = 'At least one contour includes "_experimental" indicating an unedited Limbus contour.'
        line2 = 'If a target, confirm the contour acceptability with the MD and have "_experimental" removed.'
        self.label = QtWidgets.QLabel(f'{line1}\n{line2}')
        self.layout.addWidget(self.label, 0, 0, 1, 2)

        self.checkbox = QtWidgets.QCheckBox('MD indicated contours are acceptable / Not a target', self)
        self.checkbox.stateChanged.connect(self._state_changed)
        self.layout.addWidget(self.checkbox, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

        self.button = QtWidgets.QPushButton('Close', self)
        self.button.clicked.connect(self.close)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

    def _renaming_table(self):
        df = self.backend.roi_suggestions()
        self.label = QtWidgets.QLabel(self.instructions)
        self.layout.addWidget(self.label, self.layout.rowCount() + 1, 0, 1, 2)

        self.table = QtWidgets.QTableWidget()

        rows, _ = df.shape
        self.table.setRowCount(rows)
        self.table.setColumnCount(2)

        # Fill table
        for i, row in df.iterrows():
            cell = QtWidgets.QTableWidgetItem(row['ROI Name'])
            self.table.setItem(i, 0, cell)
            if row['ROI Type'] != 'Target' and not row['Compliant']:
                selector = RestrictedScrollQComboBox(self.scrollArea)
                selector.addItem('Keep')
                for suggestion in row['Suggestions']:
                    selector.addItem(suggestion)
                self.table.setCellWidget(i, 1, selector)
                self._user_selected[row['ROI Name']] = selector
            else:
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem('---'))

        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.table.setHorizontalHeaderLabels(['ROI Name', 'Suggestions'])
        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.resizeSection(0, 100)
        horizontalHeader.resizeSection(1, 100)
        verticalHeader = self.table.verticalHeader()
        verticalHeader.setVisible(False)
        self.layout.addWidget(self.table, self.layout.rowCount() + 1, 0, 1, 2)

        self.proceed = QtWidgets.QPushButton('Proceed', self)
        self.proceed.clicked.connect(self._extract_choices)
        self.proceed.setFixedSize(100, 30)
        self.layout.addWidget(self.proceed, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

    def _confirm_table(self):
        self.layout.removeWidget(self.label)
        self.label.deleteLater()

        self.layout.removeWidget(self.table)
        self.table.deleteLater()

        self.layout.removeWidget(self.proceed)
        self.proceed.deleteLater()

        self.label = QtWidgets.QLabel('The following ROI changes will be made.\n(Red ROIs are not compliant)')
        self.layout.addWidget(self.label, 0, 0, 1, 2)

        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(len(self.roi_dict))
        self.table.setColumnCount(2)

        # Fill table
        for i, (old, new) in enumerate(self.roi_dict.items()):
            if old == new:
                old_red = QtWidgets.QLabel(old)
                old_red.setStyleSheet("color: red;")
                self.table.setCellWidget(i, 0, old_red)

                new_red = QtWidgets.QLabel(new)
                new_red.setStyleSheet("color: red;")
                self.table.setCellWidget(i, 1, new_red)
            else:
                self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(old))
                self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(new))

        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.table.setHorizontalHeaderLabels(['Old ROI Name', 'New ROI Name'])
        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.resizeSection(0, 115)
        horizontalHeader.resizeSection(1, 115)
        verticalHeader = self.table.verticalHeader()
        verticalHeader.setVisible(False)
        self.layout.addWidget(self.table, self.layout.rowCount() - 1, 0, 1, 2)

        self.categorize_checkbox = QtWidgets.QCheckBox('Attempt to categorize ROIs')
        self.layout.addWidget(self.categorize_checkbox, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_LEFT)
        self.merge_checkbox = QtWidgets.QCheckBox('Merge bilateral Femur, Lung and Kidney')
        self.layout.addWidget(self.merge_checkbox, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_LEFT)
        self.colortable_checkbox = QtWidgets.QCheckBox('Set standard colortable')
        self.layout.addWidget(self.colortable_checkbox, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_LEFT)
        self.ring_checkbox = QtWidgets.QCheckBox('Create rings')
        self.ring_checkbox.stateChanged.connect(self._ring_window)
        self.layout.addWidget(self.ring_checkbox, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_LEFT)

        self.confirm = QtWidgets.QPushButton('Confirm', self)
        self.confirm.clicked.connect(self._commit_changes)
        self.confirm.setFixedSize(100, 30)
        self.layout.addWidget(self.confirm, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)
        self.resize(self._sizeHint(0, 50))

    def _ring_window(self):
        self.layout.removeWidget(self.ring_checkbox)
        self.ring_checkbox.deleteLater()

        ring_label = QtWidgets.QLabel('Create rings:')

        self.roi_selector = QtWidgets.QComboBox()

        all_rois = self.backend.all_rois()
        all_rois.sort()

        for roi in all_rois:
            self.roi_selector.addItem(roi)

        onlyDouble = QtGui.QDoubleValidator()
        onlyDouble.setRange(0.00, 15.00, 2)

        self.inner = QtWidgets.QLineEdit()
        self.inner.setMaximumWidth(75)
        self.inner.setValidator(onlyDouble)

        self.outer = QtWidgets.QLineEdit()
        self.outer.setMaximumWidth(75)
        self.outer.setValidator(onlyDouble)

        horizontalLayout1 = QtWidgets.QHBoxLayout()
        horizontalLayout1.addWidget(ring_label, alignment=ALIGN_LEFT)
        horizontalLayout1.addWidget(self.roi_selector, alignment=ALIGN_LEFT)
        self.layout.addLayout(horizontalLayout1, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

        horizontalLayout2 = QtWidgets.QHBoxLayout()
        inner_label = QtWidgets.QLabel('Inner (cm)')
        horizontalLayout2.addWidget(inner_label, alignment=ALIGN_LEFT)
        horizontalLayout2.addWidget(self.inner, alignment=ALIGN_RIGHT)
        self.layout.addLayout(horizontalLayout2, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

        horizontalLayout3 = QtWidgets.QHBoxLayout()
        outer_label = QtWidgets.QLabel('Outer (cm)')
        horizontalLayout3.addWidget(outer_label, alignment=ALIGN_LEFT)
        horizontalLayout3.addWidget(self.outer, alignment=ALIGN_RIGHT)
        self.layout.addLayout(horizontalLayout3, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

        self.ring_button = QtWidgets.QPushButton('Create Ring', self)
        self.ring_button.clicked.connect(self._create_ring)
        self.ring_button.setFixedSize(100, 30)
        self.layout.addWidget(self.ring_button, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

        self.layout.removeWidget(self.confirm)
        self.layout.addWidget(self.confirm, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)
        self.resize(self._sizeHint(0, 125))

    def _create_ring(self, roi):
        roi = self.roi_selector.currentText()
        if self.inner.text() and self.outer.text():
            d_inner = min(max(float(self.inner.text()), 0), 15)
            d_outer = min(max(float(self.outer.text()), 0), 15)
            if d_inner < d_outer:
                self.ring_button.setText('Running...')
                QtCore.QCoreApplication.processEvents()
                _ = self.backend.create_ring(roi, d_inner, d_outer)
        self.ring_button.setText('Create Ring')

    def _extract_choices(self):
        kept = {}
        changed = {}
        for old, selector in self._user_selected.items():
            new = selector.currentText()
            if new == 'Keep':
                kept[old] = old
        for old, selector in self._user_selected.items():
            new = selector.currentText()
            if new != 'Keep':
                changed[old] = new

        kept = dict(sorted(kept.items()))
        changed = dict(sorted(changed.items()))

        self.roi_dict = {**kept, **changed}
        self._confirm_table()
        self.resize(self._sizeHint(0, 20))

    def _commit_changes(self):
        if len(self.roi_dict):
            self.backend.update_roi_names(self.roi_dict)
        self.backend.record_run('Import')

        if self.categorize_checkbox.isChecked():
            self.backend.categorize_targets()

        if self.merge_checkbox.isChecked():
            self.backend.merge_bilateral_structures()

        if self.colortable_checkbox.isChecked():
            self.backend.set_color_table()

        self.close()

    def _sizeHint(self, xpad=0, ypad=0):
        if hasattr(self, 'table'):
            h = self.table.horizontalHeader()
            v = self.table.verticalHeader()
            f = self.table.frameWidth() * 2
            total_width = max(150, min(400, h.length() + v.sizeHint().width() + f + 75))
            total_height = max(300, min(600, v.length() + h.sizeHint().height() + f + 50))
            return QtCore.QSize(total_width + xpad, total_height + ypad)

    def build(self):
        self.scrollArea = QtWidgets.QScrollArea(self)

        widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        widget.setLayout(self.layout)

        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(widget)
        self.setCentralWidget(self.scrollArea)

        self.setWindowTitle('ROI Rename')

        if self.backend._approved():
            self._approval_check()
        elif self.backend.has_experimental():
            self._experimental_check()
        else:
            self._renaming_table()
            self.resize(self._sizeHint(-10, 20))


class BeamChecking(QtWidgets.QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.backend = backend
        self.colors = []
        self.fractions_specified = False
        self.finished = False
        self.instructions = 'All tests will run automatically. Any indicated warnings should be investigated and failures require fixing before review.'
        self.setStyleSheet("""QProgressBar {background-color: #FFFFFF;
                                            border: 10px;
                                            padding: 20px;
                                            height: 20px;
                                            border-radius: 5px;
                                            text-align: center;
                                            }
                              QProgressBar::chunk {background: #BBDBED;
                                                   width:5px
                                                   }
                           """)

    def _sizeHint(self, xpad=0, ypad=0):
        if self.fractions_specified:
            ypad += 20
        if hasattr(self, 'table'):
            h = self.table.horizontalHeader()
            v = self.table.verticalHeader()
            f = self.table.frameWidth() * 2
            total_width = max(150, min(1000, h.length() + v.sizeHint().width() + f + 25))
            total_height = max(250, min(1400, v.length() + h.sizeHint().height() + f + 110))
            return QtCore.QSize(total_width + xpad, total_height + ypad)

    def _extract_and_close(self):
        self.backend.record_checking(self.statuses)
        if self.fractions_specified:
            self._set_fractions()
        if self.setup_checkbox.isChecked():
            self.backend.create_setup_beams()
        self.close()

    def _add_button(self):
        if self.finished:
            self.layout.removeWidget(self.button)
            self.button.deleteLater()
            if self.failure:
                self.button = QtWidgets.QPushButton('Quit and Fix', self)
                self.button.setStyleSheet("background-color : #f59d9d")
            else:
                self.button = QtWidgets.QPushButton('Proceed', self)
            self.button.clicked.connect(self._extract_and_close)
            self.button.setFixedSize(100, 30)
            self.layout.addWidget(self.button, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

    def _toggle_details(self):
        if self.detail_active:
            self.detail_active = False
            self.detail_color = 'light grey'
        else:
            self.detail_active = True
            self.detail_color = 'light blue'

        self.details_button.setStyleSheet(f"background-color : {self.detail_color}")

        if self.detail_active:
            fmt_text = 'The following MLCs are behind the jaws: \n'
            for issue in self.issues:
                fmt_text += str(issue)
            self.text_widget = QtWidgets.QPlainTextEdit(fmt_text)
            self.text_widget.setReadOnly(True)
            self.text_widget.setMaximumHeight(80)
            self.layout.addWidget(self.text_widget, self.layout.rowCount(), 0, 1, 2)

            self.details_button.deleteLater()
            self.layout.removeWidget(self.details_button)

            self.details_button = QtWidgets.QPushButton('Hide Details', self)
            self.details_button.clicked.connect(self._toggle_details)
            self.details_button.setFixedSize(100, 30)
            self.layout.addWidget(self.details_button, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)
            self.layout.addWidget(self.button, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)
            self.resize(self._sizeHint(5, 120))
        else:
            self.details_button.deleteLater()
            self.layout.removeWidget(self.details_button)
            self._add_details_button()
            self.text_widget.deleteLater()
            self.layout.removeWidget(self.text_widget)
            self.resize(self._sizeHint(5, 40))

    def _add_details_button(self):
        self.details_button = QtWidgets.QPushButton('Show Details', self)
        self.details_button.clicked.connect(self._toggle_details)
        self.details_button.setFixedSize(100, 30)
        self.detail_active = False
        self.layout.addWidget(self.details_button, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)
        self.resize(self._sizeHint(5, 40))

    def _setup(self):
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QtWidgets.QGridLayout()
        self.widget.setLayout(self.layout)

        self.label = QtWidgets.QLabel(self.instructions)
        self.layout.addWidget(self.label, 0, 0)

    def _beam_check_setup(self):
        self.table = QtWidgets.QTableWidget()
        self.table.verticalHeader().setVisible(False)

        self.checks = {'Jaws Gap': self.backend.min_jaw_gap,
                       'MLC Gap': self.backend.min_mlc_gap,
                       'EDW Gap': self.backend.edw_check,
                       'MLC Behind Jaws': self.backend.mlc_behind_jaws,
                       'Collimator Angles': self.backend.collimator_check,
                       'CW/CCW Check': self.backend.cw_check,
                       'Bolus Check': self.backend.bolus_check,
                       'FiF MU Limit': self.backend.fif_check,
                       'Rx Dose Limit': self.backend.rx_check,
                       'Dose Grid Size': self.backend.dose_grid_size,
                       'Iso Couch Distance': self.backend.vertical_clearance,
                       'Iso Shift Distance': self.backend.partial_arc_check,
                       'Collision Check': self.backend.collision_check,
                       'Dose Recomputation': self.backend.recompute_dose,
                       'Dose Algorithm': self.backend.algorithm_check,
                       'Dmax OAR Overlap': self.backend.dmax_check}

        self.table.setRowCount(len(self.checks))
        self.table.setColumnCount(3)

        # Column names
        self.table.setHorizontalHeaderLabels(['Check', 'Progress', 'Status'])

        self.progress_bars = {}
        self.status_values = {}

        # Fill table
        for i, check in enumerate(self.checks):
            name = QtWidgets.QTableWidgetItem(check)
            self.table.setItem(i, 0, name)

            pbar = QtWidgets.QProgressBar()
            pbar.setGeometry(10, 10, 1, 1)
            self.progress_bars[check] = pbar
            self.table.setCellWidget(i, 1, pbar)

            status = QtWidgets.QTableWidgetItem('Queued')
            status.setTextAlignment(ALIGN_CENTER)
            self.status_values[check] = status
            self.table.setItem(i, 2, status)

        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.resizeColumnsToContents()
        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.resizeSection(0, 150)
        horizontalHeader.resizeSection(1, 290)
        horizontalHeader.resizeSection(2, 290)
        self.layout.addWidget(self.table, self.layout.rowCount() + 1, 0, 1, 2)

        if self.backend.multiple_beamsets():
            self.fractions_specified = True

            label = QtWidgets.QLabel('Plan has multiple beamsets. Please give the total treatment sessions for this plan:')
            label.setStyleSheet("background-color: rgb(249, 157, 157); border: 1px solid black;")

            self.fractions = QtWidgets.QLineEdit()
            self.fractions.setMaximumWidth(50)
            onlyInt = QtGui.QIntValidator()
            onlyInt.setRange(1, 99)
            self.fractions.setValidator(onlyInt)
            self.fractions.textChanged.connect(self._add_button)

            horizontalLayout = QtWidgets.QHBoxLayout()
            horizontalLayout.addWidget(label)
            horizontalLayout.addWidget(self.fractions, alignment=ALIGN_LEFT)
            self.layout.addLayout(horizontalLayout, self.layout.rowCount(), 0, 1, 1, alignment=ALIGN_CENTER)

        self.setup_checkbox = QtWidgets.QCheckBox('Create Setup Beams (Start from A)')
        self.layout.addWidget(self.setup_checkbox, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

        self.button = QtWidgets.QPushButton('Wait...', self)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

    def check(self):
        # Run checks from backend
        self.statuses = {}

        for i, (name, function) in enumerate(self.checks.items()):
            cell = QtWidgets.QTableWidgetItem('Running')
            cell.setTextAlignment(ALIGN_CENTER)
            self.table.setItem(i, 2, cell)
            QtCore.QCoreApplication.processEvents()

            if name == 'MLC Behind Jaws':
                self.issues, status = function(self.progress_bars[name])
                if self.issues:
                    self._add_details_button()
            else:
                status = function(self.progress_bars[name])
            self.statuses[name] = status

            cell = QtWidgets.QTableWidgetItem(status.text)
            cell.setBackground(STATUS_COLORS[status.level])
            cell.setTextAlignment(ALIGN_CENTER)
            self.table.setItem(i, 2, cell)
            QtCore.QCoreApplication.processEvents()

        self.failure = False
        for status in self.statuses.values():
            if status.level == 'Fail':
                self.failure = True

        self.finished = True

        if not self.backend.multiple_beamsets() or self.fractions.text() is None:
            self._add_button()

    def _set_fractions(self):
        n_fx = int(self.fractions.text())
        self.backend.number_of_fx = n_fx
        self.backend.record_fractions(n_fx)

    def build(self):
        self._setup()
        self._beam_check_setup()
        self.setWindowTitle('Pre-MD Check')
        self.resize(self._sizeHint())


class DoseWarnings(QtWidgets.QMainWindow):
    def __init__(self, backend, request_approvals=False, setup_beams=False,
                 display_approvals=False, dosimetry=False):
        super().__init__()
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.backend = backend
        self.colors = []
        self.instructions = 'Instructions'
        self.request_approvals = request_approvals
        self.display_approvals = display_approvals
        self.dosimetry = dosimetry
        self.setup_beams = setup_beams
        self._showing_all = False

    def sizeHint(self):
        h = self.horizontalHeader()
        v = self.verticalHeader()
        f = self.frameWidth() * 2
        total_width = min(1000, h.length() + v.sizeHint().width() + f + 25)
        total_height = min(500, v.length() + h.sizeHint().height() + f)
        return QtCore.QSize(total_width, total_height)

    def _discuss_box(self, w=2):
        self.notes_label = QtWidgets.QLabel('Please use this space to discuss any warnings or failures:')
        self.layout.addWidget(self.notes_label, self.layout.rowCount() + 1, 0, 1, 2)

        self.notes = QtWidgets.QTextEdit()
        self.notes.setMinimumHeight(10)
        self.notes.setMaximumHeight(40)
        self.layout.addWidget(self.notes, self.layout.rowCount() + 1, 0, 1, w)

    def _feedback_boxes(self, w=2):
        self.feedback_label = QtWidgets.QLabel('Feel free to leave comments about the UI or any issues you encountered:')
        self.layout.addWidget(self.feedback_label, self.layout.rowCount() + 1, 0, 1, 2)

        self.feedback = QtWidgets.QTextEdit()
        self.feedback.setMinimumHeight(10)
        self.feedback.setMaximumHeight(40)
        self.layout.addWidget(self.feedback, self.layout.rowCount() + 1, 0, 1, w)

    def _toggle_table(self):
        self._showing_all = not self._showing_all
        self._update_dose_table(self.w, self.icc)

    def _update_dose_table(self, w=2, icc=False):
        if self._showing_all:
            self.show_all.setText('Hide passing')
        else:
            self.show_all.setText('Show passing')

        df, _ = self.backend.dose_warnings()

        if not self._showing_all:
            passes = df['Status'] == 'Pass'
            passes_index = np.flatnonzero(passes)
            df.drop(passes_index, inplace=True)

        if self.request_approvals or self.display_approvals:
            df.insert(2, 'Approval', None, True)

        df['Volume'] = df['Volume'] + df['Volume Units']
        df.drop(columns=['Volume Units'], inplace=True)
        df.drop(columns=['Number Of Fx'], inplace=True)

        if icc:
            df.rename(columns={'Approval': 'MD Approval'}, inplace=True)

        fill = ColorTable(df.iloc[:, 2:], allow_approval=self.request_approvals or self.display_approvals)
        self.table.setModel(fill)
        self._add_checks(df, fill)

        self.layout.addWidget(self.table, self.dose_table_row - 1, 0, 1, w, alignment=ALIGN_CENTER)
        self.resize(self.layout.sizeHint())

    def _add_checks(self, df, fill):
        if self.request_approvals or self.display_approvals:
            self.checkboxes = []
            if self.request_approvals:
                for i in range(fill.rowCount(index=0)):
                    if df.loc[df.index[i], 'Status'] == 'Pass':
                        nan = QtWidgets.QLabel('---')
                        self.checkboxes.append(nan)
                    else:
                        self.checkboxes.append(QtWidgets.QCheckBox(''))
            else:
                md_approvals = self.backend.get_approvals(df)
                for i, approved in zip(range(fill.rowCount(index=0)), md_approvals):
                    if df.loc[df.index[i], 'Status'] == 'Pass':
                        nan = QtWidgets.QLabel('---')
                        self.checkboxes.append(nan)
                    else:
                        cbox = QtWidgets.QCheckBox('')
                        self.checkboxes.append(cbox)
                        cbox.setChecked(approved)
                        cbox.setEnabled(False)

            for i, cbox in enumerate(self.checkboxes):
                if type(cbox) is QtWidgets.QCheckBox:
                    cbox.setStyleSheet("QCheckBox::indicator:unchecked{border: 1px solid red};")
                cell = QtWidgets.QWidget()
                temp_layout = QtWidgets.QHBoxLayout(cell)
                temp_layout.addWidget(cbox)
                temp_layout.setAlignment(ALIGN_CENTER)
                temp_layout.setContentsMargins(0, 0, 0, 0)
                cell.setLayout(temp_layout)
                self.table.setIndexWidget(fill.index(i, 0), cell)

    def _make_table(self, df):
        self.table = ResizableTableView()
        fill = ColorTable(df.iloc[:, 2:], allow_approval=self.request_approvals or self.display_approvals)
        self.table.setModel(fill)
        self._add_checks(df, fill)
        self.table.verticalHeader().setVisible(False)

        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.resizeColumnsToContents()

    def _dose_table(self, w=2, icc=False, replace=False):
        self.w = w
        self.icc = icc
        if self.request_approvals or self.display_approvals:
            dose_label = QtWidgets.QLabel('Below are any clinical goals and reference limits not met. Checkboxes for MD approvals.')
            self.layout.addWidget(dose_label, 0, 0, 1, w)
        else:
            dose_label = QtWidgets.QLabel('Below are any clinical goals or reference limits not met.')
            self.layout.addWidget(dose_label, 0, 0, 1, 1)

        df, fx_used = self.backend.dose_warnings()
        self.dose_df = df

        if fx_used == 'conv':
            n_fx_label = QtWidgets.QLabel('All reference dose constraints shown for conventional fractionation.')
        else:
            n_fx_label = QtWidgets.QLabel(f'All reference dose constraints shown for {fx_used} fractions.')
        bold_font = QtGui.QFont()
        bold_font.setBold(True)
        n_fx_label.setFont(bold_font)
        self.layout.addWidget(n_fx_label, 1, 0, 1, w)

        if not self._showing_all:
            passes = df['Status'] == 'Pass'
            passes_index = np.flatnonzero(passes)
            df.drop(passes_index, inplace=True)

        if self.request_approvals or self.display_approvals:
            df.insert(2, 'Approval', None, True)

        df['Volume'] = df['Volume'] + df['Volume Units']
        df.drop(columns=['Volume Units'], inplace=True)
        df.drop(columns=['Number Of Fx'], inplace=True)

        if icc:
            df.rename(columns={'Approval': 'MD Approval'}, inplace=True)

        self._make_table(df)

        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.resizeSection(1, 125)
        horizontalHeader.resizeSection(3, 75)

        self.layout.addWidget(self.table, self.layout.rowCount() + 1, 0, 1, w)
        self.dose_table_row = self.layout.rowCount()

        self.show_all = QtWidgets.QPushButton('Show passing')
        self.show_all.clicked.connect(self._toggle_table)
        self.show_all.setFixedSize(100, 30)
        self.layout.addWidget(self.show_all, 0, w-1, 2, 1, alignment=ALIGN_RIGHT)

    def _extract_comment(self):
        self.backend.write_comment(self.notes.toPlainText(), user='CMD')
        self.backend.write_comment(self.feedback.toPlainText(), user='CMD Feedback')
        self.backend.record_run('Export')
        if self.setup_beams and self.setup_checkbox.isChecked():
            self.backend.create_setup_beams()
        self.close()

    def _add_button(self):
        self.button = QtWidgets.QPushButton('Proceed', self)
        self.button.clicked.connect(self._extract_comment)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

    def _add_setup_fields(self):
        self.setup_checkbox = QtWidgets.QCheckBox('Create Setup Beams (Start from A)')
        self.layout.addWidget(self.setup_checkbox, self.layout.rowCount(), 0, 1, 2, alignment=ALIGN_CENTER)

    def _setup(self):
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QtWidgets.QGridLayout()
        self.widget.setLayout(self.layout)

    def build(self):
        self._setup()
        self._dose_table()
        self._discuss_box()
        self._feedback_boxes()
        if self.setup_beams:
            self._add_setup_fields()
        self._add_button()
        self.setWindowTitle('Pre-MD Check')
        self.resize(self.layout.sizeHint())


class MDReview(DoseWarnings):
    def __init__(self, backend):
        super().__init__(backend, True)
        self.request_approvals = False

    def _toggle_table(self):
        self._showing_all = not self._showing_all
        self._update_dose_table(self.w, self.icc)
        self._roi_renaming_table(self.w)

    def _roi_renaming_table(self, w=2, full_width=False):
        if not hasattr(self, 'label_row'):
            self.label_row = self.layout.rowCount() + 1

        rename_label = QtWidgets.QLabel('History of ROI renaming')
        self.layout.addWidget(rename_label, self.label_row, 0, 1, 1)

        if not hasattr(self, 'roi_table_row'):
            self.roi_table_row = self.layout.rowCount()

        self.rename_table = ResizableTableView()
        fill = BasicTable(self.backend.renaming_history())
        self.rename_table.setModel(fill)
        self.rename_table.resizeColumnsToContents()
        verticalHeader0 = self.rename_table.verticalHeader()
        verticalHeader0.setVisible(False)
        if full_width:
            self.layout.addWidget(self.rename_table, self.roi_table_row, 0, 1, w)
        else:
            self.layout.addWidget(self.rename_table, self.roi_table_row, 0, 1, 1)

    def _tg_263_table(self, w=2):
        compliance_label = QtWidgets.QLabel('TG-263 Compliance of all ROIs')
        self.layout.addWidget(compliance_label, self.label_row, 1, 1, 1)

        compliance_df = self.backend.roi_suggestions()
        compliance_df.drop(columns=['Suggestions', 'ROI Type'], inplace=True)

        if not self._showing_all:
            compliance_df = compliance_df[compliance_df['Compliant'] == False]

        compliance_df.loc[compliance_df['Compliant'], 'Compliant'] = 'Yes'
        compliance_df.loc[compliance_df['Compliant'] == False, 'Compliant'] = 'No'
        compliance_df.sort_values(by='Compliant')

        self.tg_table = ResizableTableView()
        fill = ColorTable(compliance_df)

        self.tg_table.setModel(fill)
        #self.tg_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.tg_table.resizeColumnsToContents()
        verticalHeader1 = self.tg_table.verticalHeader()
        verticalHeader1.setVisible(False)
        self.layout.addWidget(self.tg_table, self.roi_table_row, 1, 1, 1)

    def _cmd_note(self, w=2):
        cmd_label = QtWidgets.QLabel('Notes from dosimetry:')
        self.layout.addWidget(cmd_label, self.layout.rowCount() + 1, 0, 1, w)

        cmd_notes = self.backend.cmd_note()
        if self.backend.check_if_run('Import'):
            cmd_line = QtWidgets.QPlainTextEdit(cmd_notes)
            cmd_line.setReadOnly(True)
            cmd_line.setMinimumHeight(10)
            cmd_line.setMaximumHeight(40)
            self.layout.addWidget(cmd_line, self.layout.rowCount() + 1, 0, 1, w)
        else:
            cmd_line = QtWidgets.QLineEdit('ROI renaming script NOT yet run. Please check for non-compliant ROI names')
            cmd_line.setStyleSheet("color: red;")
            cmd_line.setReadOnly(True)
            self.layout.addWidget(cmd_line, self.layout.rowCount() + 1, 0, 1, w)

    def _extract_comment(self):
        self.backend.write_comment(self.notes.toPlainText(), user='MD')
        self.backend.write_comment(self.feedback.toPlainText(), user='MD Feedback')
        self.backend.record_run('MD')
        self.backend.record_approvals(self.checkboxes, self.dose_df)
        self.close()

    def build(self):
        self._setup()
        self._dose_table()
        self._roi_renaming_table(full_width=True)
        self._cmd_note()
        self._discuss_box()
        self._feedback_boxes()

        self.button = QtWidgets.QPushButton('Proceed', self)
        self.button.clicked.connect(self._extract_comment)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 2, alignment=ALIGN_CENTER)

        self.setWindowTitle('MD Review')
        self.resize(self.layout.sizeHint())


class ICCReview(MDReview):
    def __init__(self, backend):
        super().__init__(backend)
        self.display_approvals = False
        self.request_approvals = False

    def _toggle_table(self):
        self._showing_all = not self._showing_all
        self._update_dose_table(self.w, self.icc)
        self._roi_renaming_table(self.w)
        self._tg_263_table(self.w)
        self._beam_table()
        self.resize(self.layout.sizeHint())

    def _md_note(self, w=2):
        md_notes = self.backend.md_note()
        md_label = QtWidgets.QLabel('Notes from physician:')
        self.layout.addWidget(md_label, self.layout.rowCount() + 1, 0, 1, w)

        if self.backend.check_if_run('MD'):
            md_line = QtWidgets.QPlainTextEdit(md_notes)
            md_line.setReadOnly(True)
            md_line.setMinimumHeight(10)
            md_line.setMaximumHeight(40)
            self.layout.addWidget(md_line, self.layout.rowCount() + 1, 0, 1, w)
        else:
            md_line = QtWidgets.QLineEdit('MD did not use script during review process.')
            md_line.setStyleSheet("color: red;")
            md_line.setReadOnly(True)
            self.layout.addWidget(md_line, self.layout.rowCount() + 1, 0, 1, w)

    def _extract_comment(self):
        self.backend.write_comment(self.feedback.toPlainText(), user='ICC Feedback')
        self.close()

    def _extract_questionnaire(self):
        self.backend.write_responses(self.responses)
        self.close()

    def _beam_table(self):
        timestamp, data = self.backend.pull_beam_checking(suppress_pass=not self._showing_all)

        if timestamp is None:
            beam_label = QtWidgets.QLabel('Latest beam check results (N/A)')
            self.layout.addWidget(beam_label, self.label_row, 2, 1, 1)

            self.beam_table = QtWidgets.QTableWidget()
            self.beam_table.setColumnCount(1)

            if data:
                self.beam_table.setRowCount(len(data))
            else:
                self.beam_table.setRowCount(0)

            cell = QtWidgets.QTableWidgetItem('Pre-MD beam check not run')
            cell.setBackground(STATUS_COLORS['Fail'])
            cell.setTextAlignment(ALIGN_CENTER)
            self.beam_table.setItem(0, 0, cell)

            if data:
                verticalHeader = self.beam_table.verticalHeader
                verticalHeader.setVisible(False)
            self.beam_table.setHorizontalHeaderLabels(['Status'])
            horizontalHeader = self.beam_table.horizontalHeader()
            horizontalHeader.resizeSection(0, 290)

            if not hasattr(self, 'beam_table_row'):
                self.beam_table_row = self.layout.rowCount() - 1

            self.layout.addWidget(self.beam_table, self.beam_table_row, 2, 1, 1)

        else:
            timestamp_fmt, _ = timestamp.rsplit('-', 1)
            timestamp_fmt = timestamp_fmt.replace('T', ' @ ').replace('.', ':')

            beam_label = QtWidgets.QLabel(f'Latest beam check results ({timestamp_fmt})')
            self.layout.addWidget(beam_label, self.label_row, 2, 1, 1)

            self.beam_table = QtWidgets.QTableWidget()
            self.beam_table.setRowCount(len(data))
            self.beam_table.setColumnCount(2)

            self.beam_table.setHorizontalHeaderLabels(['Check', 'Status'])
            for i, (check, result) in enumerate(data.items()):
                name = QtWidgets.QTableWidgetItem(check)
                self.beam_table.setItem(i, 0, name)

                status = QtWidgets.QTableWidgetItem(result.text)
                status.setBackground(STATUS_COLORS[result.level])
                status.setTextAlignment(ALIGN_CENTER)
                self.beam_table.setItem(i, 1, status)

            self.beam_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            self.beam_table.resizeColumnsToContents()
            verticalHeader = self.beam_table.verticalHeader()
            verticalHeader.setVisible(False)
            horizontalHeader = self.beam_table.horizontalHeader()
            horizontalHeader.resizeSection(0, 133)
            horizontalHeader.resizeSection(2, 350)

            if not hasattr(self, 'beam_table_row'):
                self.beam_table_row = self.layout.rowCount() - 1

            self.layout.addWidget(self.beam_table, self.beam_table_row, 2, 1, 1)
            self.resize(100, 100)

    def _create_questionnaire(self):
        questions = self.backend.questionnaire()
        self.responses = {}

        for i, question in enumerate(questions):
            q_label = QtWidgets.QLabel(question)
            self.layout.addWidget(q_label, self.layout.rowCount() + 1, 0, 1, 3)
            self.responses[i] = QtWidgets.QLineEdit()
            self.layout.addWidget(self.responses[i], self.layout.rowCount() + 1, 0, 1, 3)

        self.button.clicked.connect(self._extract_questionnaire)
        self.setWindowTitle('ICC Questionnaire')

    def build(self):
        self._setup()
        self._dose_table(w=3, icc=True)
        self._roi_renaming_table(w=3)
        self._tg_263_table(w=3)
        self._beam_table()
        self._cmd_note(w=3)
        self._md_note(w=3)
        self._discuss_box(w=3)
        self._feedback_boxes(w=3)

        # List Notes from MD and CMD
        self.button = QtWidgets.QPushButton('Proceed', self)
        self.button.clicked.connect(self._extract_comment)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 3, alignment=ALIGN_CENTER)

        self.setWindowTitle('ICC Review')
        self.resize(self.layout.sizeHint())


class Selector(QtWidgets.QMainWindow):
    def __init__(self, ui):
        super().__init__()
        self.ui = ui

        self.scrollArea = QtWidgets.QScrollArea(self)

        widget = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout()
        widget.setLayout(self.layout)

        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(widget)
        self.setCentralWidget(self.scrollArea)

        self.selector = RestrictedScrollQComboBox(self.scrollArea)
        self.selector.addItem('On Import')
        self.selector.addItem('Pre-MD')
        self.selector.addItem('On Export')
        self.selector.addItem('MD Review')
        self.selector.addItem('ICC Review')

        self.layout.addWidget(self.selector, 1, 0, 1, 2)

        button = QtWidgets.QPushButton('Run', self)
        button.clicked.connect(self._run)
        button.setFixedSize(100, 30)
        self.layout.addWidget(button, 2, 0, 1, 2, alignment=ALIGN_CENTER)

    def _run(self):
        self.choice = self.selector.currentText()
        self.close()


class CombinedExport(QtWidgets.QMainWindow):
    def __init__(self, backend):
        super().__init__()
        self.setWindowFlag(QtCore.Qt.WindowType.WindowStaysOnTopHint, True)
        self.backend = backend
        self.expanded = False
        self.colors = []
        self.target_cboxes = {}
        self.drr_cboxes = {}
        self.poi_cboxes = {}
        self.instructions = 'Select any IGRT ROIs to be sent to Aria in the table.\nAll Targets will automatically send to Aria.\nAll ROIs will be sent to MIM and Mobius.'
        self.operations = ['ARIA', 'MIM', 'Mobius', 'AlignRT', 'Generate QA Plan']
        self.viewed_options = {'Export Targets': False,
                               'Export POIs': False,
                               'BEV ROIs': False}

    def _sizeHint(self, xpad=0, ypad=0):
        if hasattr(self, 'table'):
            h = self.table.horizontalHeader()
            v = self.table.verticalHeader()
            f = self.table.frameWidth() * 2
            total_width = max(50, min(500, h.length() + v.sizeHint().width() + f))
            total_height = max(50, min(300, v.length() + h.sizeHint().height() + f))
            return QtCore.QSize(total_width + xpad, total_height + ypad)

    def _setup(self):
        self.layout = None
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.layout = QtWidgets.QGridLayout()
        self.widget.setLayout(self.layout)

        self.label = QtWidgets.QLabel(self.instructions, alignment=ALIGN_CENTER)
        self.layout.addWidget(self.label, 0, 0, 2, 3)

    def _progress_build(self):
        self.instructions = 'Please wait while each step runs.\nLocations must be checked for proper export.'

        if self.expanded:
            self.layout.removeWidget(self.target_table)
            self.target_table.deleteLater()

        self._setup()

        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(len(self.operations))
        self.table.setColumnCount(2)
        self.resize(self._sizeHint())

        self.table.setHorizontalHeaderLabels(['Export Step', 'Progress'])

        for i, name in enumerate(self.operations):
            cell = QtWidgets.QLabel(name)
            self.table.setCellWidget(i, 0, cell)

            status = QtWidgets.QTableWidgetItem('Queued')
            status.setBackground(STATUS_COLORS['Warn'])
            status.setTextAlignment(ALIGN_CENTER)
            self.table.setItem(i, 1, status)

        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.resizeSection(0, 117)
        horizontalHeader.resizeSection(1, 117)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.layout.addWidget(self.table, self.layout.rowCount() + 1, 0, 1, 3, alignment=ALIGN_CENTER)

        self.button = QtWidgets.QPushButton('Wait...', self)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount() + 1, 0, 1, 3, alignment=ALIGN_CENTER)

        self.resize(self._sizeHint(75, 100))
        self._run_exports()

    def _run_exports(self):
        QtCore.QCoreApplication.processEvents()
        for i, name in enumerate(self.operations):
            status = QtWidgets.QTableWidgetItem('Running')
            status.setBackground(STATUS_COLORS['Warn'])
            status.setTextAlignment(ALIGN_CENTER)
            self.table.setItem(i, 1, status)
            QtCore.QCoreApplication.processEvents()

            skipped = True
            failed = False
            versa_warning = False
            alignrt_warning = False

            roi_visible = self.backend.visible_rois()
            poi_visible = self.backend.visible_pois()

            self.backend.update_roi_dose_statistics()

            if name == 'AlignRT':
                if self.alignrt_export.isChecked():
                    if 'External Vision RT' not in self.backend.igrt_eligible():
                        alignrt_warning = True
                    else:
                        self.backend.export(location=name,
                                            igrt_structures='External Vision RT')
                    skipped = False
            elif name == 'ARIA':
                if self.aria_export.isChecked():
                    igrt_selected = [roi for (roi, box) in self.roi_igrt.items() if box.isChecked()]
                    targets_to_exclude = None
                    if len(self.target_cboxes):
                        targets_to_exclude = [tar for (tar, box) in self.target_cboxes.items() if not box.isChecked()]

                    if len(self.drr_cboxes):
                        drrs_to_include = [roi for (roi, box) in self.drr_cboxes.items() if box.isChecked()]
                        self.backend.set_roi_poi_visibility(rois=drrs_to_include, pois=None)

                    versa_warning = self.backend.versa_roi_warning(targets_to_exclude, igrt_selected)

                    pois_to_exclude = [poi for (poi, box) in self.poi_cboxes.items() if not box.isChecked()]

                    self.backend.export(location=name,
                                        mv_ports=self.mv_port.isChecked(),
                                        drr_setting=self.drr_selector.currentText(),
                                        igrt_structures=igrt_selected,
                                        targets_to_exclude=targets_to_exclude,
                                        pois_to_exclude=pois_to_exclude,
                                        machine_alias=self.alias_selector.currentText())

                    self.backend.set_roi_poi_visibility(rois=roi_visible, pois=poi_visible)

                    skipped = False
            elif name == 'MIM':
                if self.mim_export.isChecked():
                    self.backend.export(location=name)
                    skipped = False
            elif name == 'Mobius3D':
                if self.mobius_export.isChecked():
                    self.backend.export(location=name,
                                        machine_alias=self.alias_selector.currentText())
                    skipped = False
            elif name == 'Generate QA Plan':
                if self.generate_qa.isChecked():
                    self.backend.create_qa_plan()
                    skipped = False

            if failed:
                status = QtWidgets.QTableWidgetItem('Failed')
                status.setBackground(STATUS_COLORS['Fail'])
            elif versa_warning:
                status = QtWidgets.QTableWidgetItem('Versa ROIs > 10')
                status.setBackground(STATUS_COLORS['Warn'])
            elif alignrt_warning:
                status = QtWidgets.QTableWidgetItem('No "External Vision RT"')
                status.setBackground(STATUS_COLORS['Warn'])
            elif skipped:
                status = QtWidgets.QTableWidgetItem('Skipped')
                status.setBackground(STATUS_COLORS['Pass'])
            else:
                status = QtWidgets.QTableWidgetItem('Complete')
                status.setBackground(STATUS_COLORS['Pass'])
            status.setTextAlignment(ALIGN_CENTER)
            self.table.setItem(i, 1, status)
            QtCore.QCoreApplication.processEvents()

        self.button.deleteLater()
        self.layout.removeWidget(self.button)

        self.button = QtWidgets.QPushButton('Close', self)
        self.button.clicked.connect(self.close)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount()-1, 0, 1, 3, alignment=ALIGN_CENTER)

    def _side_settings(self):
        self.drr_label = QtWidgets.QLabel('DRR Settings')
        drr_options = self.backend.drr_options()
        self.drr_selector = QtWidgets.QComboBox()
        for drr_option in drr_options:
            self.drr_selector.addItem(drr_option)

        self.alias_label = QtWidgets.QLabel('Machine Alias')
        alias_options = self.backend.alias_options()
        self.alias_selector = QtWidgets.QComboBox()
        for alias_option in alias_options:
            self.alias_selector.addItem(alias_option)

        alias_value = self.backend.alias_value()
        if alias_value:
            index = alias_options.index(alias_value)
            self.alias_selector.setCurrentIndex(index)

        self.mv_port = QtWidgets.QCheckBox('Include MV Ports')
        self.mv_port.setChecked(False)
        self.alignrt_export = QtWidgets.QCheckBox('Export to Align RT')
        self.alignrt_export.setChecked(False)

        self.aria_export = QtWidgets.QCheckBox('Export to Aria')
        self.aria_export.setChecked(True)
        self.mim_export = QtWidgets.QCheckBox('Export to MIM')
        self.mim_export.setChecked(True)
        self.mobius_export = QtWidgets.QCheckBox('Export to Mobius')
        self.mobius_export.setChecked(True)
        self.generate_qa = QtWidgets.QCheckBox('Generate QA Plan')
        self.generate_qa.setChecked(True)

        self.layout.addWidget(self.drr_label, 3, 0, 1, 1, alignment=ALIGN_LEFT)
        self.layout.addWidget(self.drr_selector, 3, 1, 1, 1, alignment=ALIGN_LEFT)

        self.layout.addWidget(self.alias_label, 4, 0, 1, 1, alignment=ALIGN_LEFT)
        self.layout.addWidget(self.alias_selector, 4, 1, 1, 1, alignment=ALIGN_LEFT)

        self.layout.addWidget(self.mv_port, 6, 0, 1, 2, alignment=ALIGN_LEFT)
        self.layout.addWidget(self.alignrt_export, 7, 0, 1, 2, alignment=ALIGN_LEFT)

        self.layout.addWidget(self.aria_export, 9, 0, 1, 2, alignment=ALIGN_LEFT)
        self.layout.addWidget(self.mim_export, 10, 0, 1, 2, alignment=ALIGN_LEFT)
        self.layout.addWidget(self.mobius_export, 11, 0, 1, 2, alignment=ALIGN_LEFT)
        self.layout.addWidget(self.generate_qa, 12, 0, 1, 2, alignment=ALIGN_LEFT)

    def _all_igrt(self):
        for cbox in self.roi_igrt.values():
            cbox.setChecked(self._igrt_toggle)
        if self._igrt_toggle:
            self.toggle.setText('Unselect All')
        else:
            self.toggle.setText('Select All')
        self._igrt_toggle = not self._igrt_toggle

    def _igrt_selection(self):
        self.roi_igrt = {}
        self._igrt_toggle = True
        rois = self.backend.igrt_eligible()

        self.toggle = QtWidgets.QPushButton('Select All')
        self.toggle.clicked.connect(self._all_igrt)

        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(len(rois))
        self.table.setColumnCount(2)

        self.table.setHorizontalHeaderLabels(['ROI Name', 'Export to Aria'])

        igrt_probable = self.backend.igrt_probable()

        for i, roi in enumerate(rois):
            roi_cell = QtWidgets.QTableWidgetItem(roi)
            checkbox_cell = QtWidgets.QCheckBox()
            checkbox_cell.setStyleSheet("margin-left:30%; margin-right:70%; text-align: center;")

            if roi in igrt_probable:
                checkbox_cell.setChecked(True)

            self.roi_igrt[roi] = checkbox_cell
            self.table.setItem(i, 0, roi_cell)
            self.table.setCellWidget(i, 1, checkbox_cell)

        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.table.verticalHeader().setVisible(False)
        horizontalHeader = self.table.horizontalHeader()
        horizontalHeader.resizeSection(0, 150)
        horizontalHeader.resizeSection(1, 80)
        self.layout.addWidget(self.table, 3, 2, self.layout.rowCount() - 4, 1)
        self.layout.addWidget(self.toggle, self.layout.rowCount() - 1, 2, 1, 1, alignment=ALIGN_CENTER)

    def _extra_options(self):
        self.expanded = True
        self.layout.removeWidget(self.exclude_button)
        self.exclude_button.deleteLater()

        self.options_label = QtWidgets.QLabel('Select additional options table\nApplies to ARIA-PROD ONLY', alignment=ALIGN_CENTER)
        self.layout.addWidget(self.options_label, 0, 3, 1, 2)

        self.options_selector = QtWidgets.QComboBox()

        for table in ['Export Targets', 'Export POIs', 'BEV ROIs']:
            self.options_selector.addItem(table)

        self.options_selector.setFixedSize(150, 30)
        self.layout.addWidget(self.options_selector, 1, 3, 2, 2, alignment=ALIGN_CENTER)

        self.options_selector.currentTextChanged.connect(self._options_distributor)

        self._targets_table()
        self.viewed_options['Export Targets'] = True

        self.resize(self._sizeHint(450, 50))
        QtCore.QCoreApplication.processEvents()

    def _options_distributor(self):
        selected = self.options_selector.currentText()

        if selected == 'Export Targets':
            self._targets_table()
        elif selected == 'Export POIs':
            self._poi_table()
        elif selected == 'BEV ROIs':
            self._bev_table()
        self.viewed_options[selected] = True

    def _poi_table(self):
        pois = [r.Name for r in self.backend.case.PatientModel.PointsOfInterest]

        self.poi_table = QtWidgets.QTableWidget()
        self.poi_table.setRowCount(len(pois))
        self.poi_table.setColumnCount(2)
        self.poi_table.setHorizontalHeaderLabels(['POI Name', 'Export to Aria'])

        for i, poi in enumerate(pois):
            poi_cell = QtWidgets.QTableWidgetItem(poi)
            self.poi_table.setItem(i, 0, poi_cell)
            if not self.viewed_options['Export POIs']:
                checkbox_cell = QtWidgets.QCheckBox()
                checkbox_cell.setStyleSheet("margin-left:30%; margin-right:70%; text-align: center;")
                checkbox_cell.setChecked(True)
                self.poi_cboxes[poi] = checkbox_cell
            self.poi_table.setCellWidget(i, 1, self.poi_cboxes[poi])

        self.poi_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.poi_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.poi_table.resizeColumnsToContents()
        self.poi_table.verticalHeader().setVisible(False)

        horizontalHeader = self.poi_table.horizontalHeader()
        horizontalHeader.resizeSection(0, 100)
        horizontalHeader.resizeSection(1, 80)
        self.layout.addWidget(self.poi_table, 3, 3, self.layout.rowCount() - 4, 1)

    def _bev_table(self):
        rois = [r.Name for r in self.backend.case.PatientModel.RegionsOfInterest]

        self.drr_table = QtWidgets.QTableWidget()
        self.drr_table.setRowCount(len(rois))
        self.drr_table.setColumnCount(2)
        self.drr_table.setHorizontalHeaderLabels(['ROI Name', 'Show in DRR'])

        for i, roi in enumerate(rois):
            roi_cell = QtWidgets.QTableWidgetItem(roi)
            self.drr_table.setItem(i, 0, roi_cell)
            if not self.viewed_options['BEV ROIs']:
                checkbox_cell = QtWidgets.QCheckBox()
                checkbox_cell.setStyleSheet("margin-left:30%; margin-right:70%; text-align: center;")
                checkbox_cell.setChecked(False)
                self.drr_cboxes[roi] = checkbox_cell
            self.drr_table.setCellWidget(i, 1, self.drr_cboxes[roi])

        self.drr_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.drr_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.drr_table.resizeColumnsToContents()
        self.drr_table.verticalHeader().setVisible(False)
        horizontalHeader = self.drr_table.horizontalHeader()
        horizontalHeader.resizeSection(0, 100)
        horizontalHeader.resizeSection(1, 80)
        self.layout.addWidget(self.drr_table, 3, 3, self.layout.rowCount() - 4, 1)

    def _targets_table(self):
        targets = self.backend.targets_eligible()

        self.target_table = QtWidgets.QTableWidget()
        self.target_table.setRowCount(len(targets))
        self.target_table.setColumnCount(2)
        self.target_table.setHorizontalHeaderLabels(['Target Name', 'Export to Aria'])

        for i, target in enumerate(targets):
            target_cell = QtWidgets.QTableWidgetItem(target)
            self.target_table.setItem(i, 0, target_cell)
            if not self.viewed_options['Export Targets']:
                print('cretaing target cboxes')
                checkbox_cell = QtWidgets.QCheckBox()
                checkbox_cell.setStyleSheet("margin-left:30%; margin-right:70%; text-align: center;")
                checkbox_cell.setChecked(True)
                self.target_cboxes[target] = checkbox_cell
            self.target_table.setCellWidget(i, 1, self.target_cboxes[target])

        self.target_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.target_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.target_table.resizeColumnsToContents()
        self.target_table.verticalHeader().setVisible(False)
        horizontalHeader = self.target_table.horizontalHeader()
        horizontalHeader.resizeSection(0, 100)
        horizontalHeader.resizeSection(1, 80)
        self.layout.addWidget(self.target_table, 3, 3, self.layout.rowCount() - 4, 1)

    def _export_button(self):
        self.button = QtWidgets.QPushButton('Export', self)
        self.button.clicked.connect(self._progress_build)
        self.button.setFixedSize(100, 30)
        self.layout.addWidget(self.button, self.layout.rowCount(), 0, 1, 3, alignment=ALIGN_CENTER)

    def _exclude_button(self):
        self.exclude_button = QtWidgets.QPushButton('Extra Options', self)
        self.exclude_button.clicked.connect(self._extra_options)
        self.exclude_button.setFixedSize(100, 30)
        self.layout.addWidget(self.exclude_button, self.layout.rowCount(), 0, 1, 3, alignment=ALIGN_CENTER)

    def build(self):
        self._setup()
        self._side_settings()
        self._igrt_selection()
        self._exclude_button()
        self._export_button()
        self.setWindowTitle('Exporter')
        self.resize(self._sizeHint(225, 50))


class UserInterface:
    def __init__(self, backend: BackendManager) -> None:
        self.backend = backend
        self.app = QtWidgets.QApplication(sys.argv)

    def for_testing(self):
        window = Selector(ui=self)
        window.show()
        self.app.exec()

        if window.choice == 'On Import':
            self.on_import()
        elif window.choice == 'Pre-MD':
            self.on_pre_MD()
        elif window.choice == 'On Export':
            self.on_export()
        elif window.choice == 'MD Review':
            self.on_MD()
        elif window.choice == 'ICC Review':
            self.on_ICC()

    def on_import(self):
        window = ROIRenaming(self.backend)
        window.build()
        window.show()
        sys.exit(self.app.exec())

    def on_pre_MD(self):
        if not self.backend.check_if_run('Import'):
            window = ROIRenaming(self.backend)
            window.build()
            window.show()
            self.app.exec()

            if self.backend._approved():
                return None

        beam_window = BeamChecking(self.backend)
        beam_window.build()
        beam_window.show()
        beam_window.check()
        self.app.exec()

        if not beam_window.failure:
            setup_checked = not beam_window.setup_checkbox.isChecked()
            dose_window = DoseWarnings(self.backend, setup_beams=setup_checked, dosimetry=True)
            dose_window.build()
            dose_window.show()
            sys.exit(self.app.exec())

    def on_export(self):
        export_window = CombinedExport(self.backend)
        export_window.build()
        export_window.show()
        sys.exit(self.app.exec())

    def on_MD(self):
        window = MDReview(self.backend)
        window.build()
        window.show()
        sys.exit(self.app.exec())

    def on_ICC(self):
        window = ICCReview(self.backend)
        window.build()
        window.show()
        sys.exit(self.app.exec())
