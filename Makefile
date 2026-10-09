.PHONY: help demo api test eval-recall eval-attacks clean

PYTHON ?= python
RUN := PYTHONPATH=. $(PYTHON)

help:
	@echo "常用命令："
	@echo "  make demo          跑端到端四个场景"
	@echo "  make api           启动 FastAPI 服务 (localhost:8000/docs)"
	@echo "  make ui            启动 Gradio 界面 (localhost:7860)"
	@echo "  make test          跑测试套件"
	@echo "  make eval-recall   检索评测 + 融合权重网格搜索"
	@echo "  make eval-attacks  护栏拦截率 / 误拒率"
	@echo "  make expand        按模板变异扩充攻击样本集"
	@echo "  make clean         清理缓存与本地索引"

demo:
	$(RUN) demo.py

api:
	$(RUN) -m app.api

ui:
	$(RUN) -m app.ui

test:
	$(RUN) -m unittest discover -s tests -v

eval-recall:
	$(RUN) eval/recall.py

eval-attacks:
	$(RUN) eval/attacks.py --sweep

expand:
	$(RUN) tools/expand_attacks.py --target 820 --indirect-ratio 0.4

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -f data/index.db
