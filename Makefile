.PHONY: venv
venv:
	source .venv/Scripts/activate

.PHONY: install torch-gpu
torch-gpu:
	pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

.PHONY: llama3
llama3:
	python llama3.py

.PHONY: gpt-j
gpt-j:
	python gpt_j.py

.PHONY: flan
flan:
	python flan_t5_large.py

.PHONY: phi-mini
phi:
	python phi_mini.py