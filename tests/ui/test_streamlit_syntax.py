import pathlib
import py_compile


def test_streamlit_compiles():
    path = pathlib.Path("apps/workspace-ui/streamlit_app.py")
    py_compile.compile(str(path), doraise=True)
