@echo off
taskkill /F /IM python.exe
timeout /t 2 /nobreak >nul
del test_logic.py qa_runner_100.py qa_runner_100_mock.py qa_runner_mini.py check_lite_model.py check_models.py check_models_file.py debug_db_connection.py debug_db_v2.py debug_hardcoded.py debug_password.py debug_schema_v2.py debug_stats.py debug_user_login.py fast_qa.py final_verify.py /F /Q
del list_test_students.py simple_verify.py simulate_login.py test_fixes.py test_gemini.py test_py.py test_suite.py verify_model_access.py verify_raw_gemini.py verify_v2.py run_server_debug.bat server.log ai_engine_imports.py /F /Q
del project_summary.md walkthrough_v2.md review_summary.md SETUP_GUIDE.md QA_Report.md implementation_plan.md implementation_plan_v2.md plan-template.md springboot-vibe-coding-lab.md TESTING.md /F /Q
del cleanup.py /F /Q
echo Done > cleanup_status.txt
