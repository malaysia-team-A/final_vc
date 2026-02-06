@echo on
move /y Chatbot_TestData.xlsx data\
move /y *.json data\
move /y *.md docs\
move /y knowledge_base data\
move /y *.csv data\
del cleanup_files.bat
