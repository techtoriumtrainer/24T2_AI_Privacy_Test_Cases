# 24T2_AI_Privacy_Test_Cases
Command to handle Execution Policy (if needed)
Set-ExecutionPolicy RemoteSigned     
Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser     
Command to create Virtual Environment
python -m venv .venv 
Command to Activate virtual environment
.venv\scripts\activate
Command to all install dependencies through requirements.txt file
pip install -r requirements.txt
Command to run the file
python filename.py
Command to run test case
pytest -v
