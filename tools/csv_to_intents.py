import csv
import yaml


class IndentDumper(yaml.Dumper):
    """自定义 YAML Dumper，增加列表缩进"""
    def increase_indent(self, flow=False, indentless=False):
        return super(IndentDumper, self).increase_indent(flow, False)


intents = []

with open("Semantic.csv", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

for row in rows[1:]:
    intent = row[2]
    if not intent:
        continue

    nlp_code = row[4]
    slot_desc = row[5]

    slots = []
    if slot_desc:
        slot = {}
        slot["name"] = "value"
        slot["alias"] = "v"
        slot["type"] = "int"

        if '{"v":0}' in slot_desc:
            slot["fixed"] = 0
            slot["required"] = True
        else:
            if "100" in slot_desc:
                slot["range"] = "0-100"
                slot["unit"] = "percent"
            elif "10" in slot_desc:
                slot["range"] = "0-10"
                slot["unit"] = "level"
            slot["required"] = False

        slots.append(slot)

    intents.append({
        "name": intent,
        "description": intent,
        "nlp_code": nlp_code,
        "app_code": nlp_code,
        "slots": slots
    })

doc = {
    "domain": "control",
    "description": "系统控制相关意图",
    "intents": intents
}

with open("intents/control.yaml", "w", encoding="utf-8") as f:
    yaml.dump(doc, f, Dumper=IndentDumper, allow_unicode=True, sort_keys=False, default_flow_style=False)
