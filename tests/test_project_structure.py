import os

def test_archivo_run_existe():
    assert os.path.exists("run.py")

def test_carpeta_app_existe():
    assert os.path.isdir("app")

def test_carpeta_routes_existe():
    assert os.path.isdir("app/routes")

def test_carpeta_services_existe():
    assert os.path.isdir("app/services")

def test_carpeta_utils_existe():
    assert os.path.isdir("app/utils")