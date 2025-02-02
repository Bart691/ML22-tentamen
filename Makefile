
.DEFAULT: run

run_linear:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/01_model_design.py

run_gru:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/01D_gru_model_design.py

run_gruatt:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/01D_gruatt_model_design.py

run_bestmodel:	
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/2C_model_design.py

tune: 
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/02_tune.py

result: 
	export PYTHONPATH="$$HOME/code/tentamen" ;poetry run python dev/scripts/03_result.py

format:
	poetry run isort dev
	poetry run black dev
	poetry run isort tentamen
	poetry run black tentamen

lint:
	poetry run flake8 dev
	poetry run flake8 tentamen
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports dev
	poetry run mypy --no-strict-optional --warn-unreachable --show-error-codes --ignore-missing-imports tentamen


clean:
	rm -rf logs/*
	rm -rf models/*


