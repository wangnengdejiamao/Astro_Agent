Set shell = CreateObject("WScript.Shell")
shell.CurrentDirectory = "C:\Users\Administrator\Desktop\rag\prompt2graph_for_astronomy"
shell.Run """C:\ProgramData\anaconda3\python.exe"" ""C:\Users\Administrator\Desktop\rag\prompt2graph_for_astronomy\run_frontend_server.py""", 0, False
