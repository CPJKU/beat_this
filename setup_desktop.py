"""Verify source runtime and create a fresh shortcut for the current location."""
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
os.chdir(root)
import torch
from beat_this.preprocessing import LogMelSpect
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from PySide6.QtWidgets import QApplication

LogMelSpect()(torch.zeros(22050))
DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=50)
app = QApplication([])
from beat_this.desktop import MainWindow
window = MainWindow()
window.close()
environment = os.environ.copy()
environment['BPM_APP_ROOT'] = str(root)
script = r'''
$taskRoot = $env:BPM_APP_ROOT
$taskShell = New-Object -ComObject WScript.Shell
$taskLink = $taskShell.CreateShortcut((Join-Path $taskRoot 'BPM Finder.lnk'))
$taskLink.TargetPath = Join-Path $taskRoot '.venv-desktop\Scripts\pythonw.exe'
$taskLink.Arguments = '"' + (Join-Path $taskRoot 'launch_ui.pyw') + '"'
$taskLink.WorkingDirectory = $taskRoot
$taskLink.Description = 'BPM Finder'
$taskLink.Save()
'''
subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
               env=environment, check=True)
print('PASS: CPU runtime, audio preprocessing, DBN, UI and shortcut verified.')
