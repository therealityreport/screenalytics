import compileall


def test_ui_file_compiles():
    assert compileall.compile_file("apps/workspace-ui/streamlit_app.py", quiet=1)
