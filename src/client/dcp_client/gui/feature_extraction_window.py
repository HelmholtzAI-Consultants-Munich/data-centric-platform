from qtpy.QtWidgets import QDialog, QGridLayout, QCheckBox, QLineEdit, QFileDialog, QLabel, QPushButton
from PyQt5.QtCore import Qt

class ExtractFeaturesDialog(QDialog):
    """Dialog to select features and filename for extraction."""

    def __init__(self, parent=None, default_filename: str = "extracted_features.csv"):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Extract features")
        self.setStyleSheet("background-color: #f3f3f3;")
        layout = QGridLayout()
        

        lbl = QLabel("Include features")
        layout.addWidget(lbl, 0, 0, 1, 3)

        # Feature checkboxes
        self.cb_area = QCheckBox("Area")
        self.cb_perim = QCheckBox("Perimeter")
        self.cb_mean = QCheckBox("Intensity Mean")
        self.cb_std = QCheckBox("Intensity std")
        for cb in (self.cb_area, self.cb_perim, self.cb_mean, self.cb_std):
            cb.setChecked(True)

        layout.addWidget(self.cb_area, 1, 0)
        layout.addWidget(self.cb_perim, 1, 1)
        layout.addWidget(self.cb_mean, 2, 0)
        layout.addWidget(self.cb_std, 2, 1)

        # Filename field + browse
        self.filename_edit = QLineEdit(default_filename)
        self.browse_btn = QPushButton("Save as...")
        self.browse_btn.setStyleSheet(
            """QPushButton 
            { 
                  background-color: #3d81d1;
                  font-size: 10px; 
                  font-weight: bold;
                  color: #ffffff; 
                  border-radius: 5px;
                  padding: 8px 16px; }"""
            "QPushButton:hover { background-color: #7bc432; }"
          
        )
        self.browse_btn.clicked.connect(self.on_browse)
        layout.addWidget(self.filename_edit, 3, 0, 1, 2)
        layout.addWidget(self.browse_btn, 3, 2)

        self.setLayout(layout)

    def on_browse(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save features as",
            self.filename_edit.text(),
            "CSV Files (*.csv);;All files (*)"
        )

        # User pressed Cancel → do nothing
        if not path:
            return

        # User pressed Save
        self.filename_edit.setText(path)

        selected = {
            "Area [pix^2]": self.cb_area.isChecked(),
            "Perimeter [pix]": self.cb_perim.isChecked(),
            "Intensity Mean": self.cb_mean.isChecked(),
            "Intensity std": self.cb_std.isChecked(),
        }

        if hasattr(self.parent, "app") and hasattr(self.parent.app, "extract_features"):
            ok, msg = self.parent.app.extract_features(path, selected)

            if hasattr(self.parent, 'create_warning_box'):
                title = "Information" if ok else "Warning"
                self.parent.create_warning_box(msg, message_title=title)

            if ok:
                self.accept()
            else:
                self.reject()
        else:
            if hasattr(self.parent, 'create_warning_box'):
                self.parent.create_warning_box(
                    "Feature extraction is not available.",
                    message_title="Error"
                )
            self.reject()