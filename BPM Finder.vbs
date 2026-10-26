Set shell = CreateObject("WScript.Shell")
Set files = CreateObject("Scripting.FileSystemObject")
root = files.GetParentFolderName(WScript.ScriptFullName)
python = files.BuildPath(root, ".venv-desktop\Scripts\pythonw.exe")
launcher = files.BuildPath(root, "launch_ui.pyw")
If Not files.FileExists(python) Then
    MsgBox "Please run Setup UI.cmd first.", 48, "BPM Finder"
Else
    shell.CurrentDirectory = root
    shell.Run Chr(34) & python & Chr(34) & " " & Chr(34) & launcher & Chr(34), 0, False
End If
