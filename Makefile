VENV_DIR?="venv"
USER_UID=$(shell id -u)


new-venv:
	echo "Creating new virtual environment at $(VENV_DIR)"
	mkdir $(VENV_DIR)
	cd $(VENV_DIR)
	python3 -m venv $(VENV_DIR)

