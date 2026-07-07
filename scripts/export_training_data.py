from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.reminder_service import ReminderService
from app.storage import JSONReminderStorage
from app.tool_executor import ToolExecutor
from app.tool_registry import get_tools

SYSTEM_PROMPT = (
    "你是小暖同学，一位温暖、耐心的长者陪伴助手。"
    "你可以使用提醒工具。只有在用户明确要求创建、查询、更新、删除提醒时才调用工具。"
    "普通聊天不要调用工具。不要伪造工具结果，只有看到 role=tool 的返回后，才能告诉用户操作成功。"
)


class DataBuilder:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.storage = JSONReminderStorage(storage_path)
        self.service = ReminderService(self.storage)
        self.executor = ToolExecutor(self.service)
        self.tools = get_tools()
        self.call_idx = 1

    def reset_storage(self) -> None:
        self.storage_path.write_text("[]", encoding="utf-8")

    def _next_call_id(self) -> str:
        call_id = f"call_{self.call_idx:04d}"
        self.call_idx += 1
        return call_id

    def _tool_call_msg(self, tool_name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                }
            ],
        }

    def _assistant_reply(self, tool_name: str, payload: dict[str, Any], user_text: str) -> str:
        status = payload.get("status")
        if status == "success":
            if tool_name == "create_reminder":
                reminder = payload.get("reminder", {})
                return f"好的，已经为您设置提醒：{reminder.get('time_text', '')}{reminder.get('task', '')}。"
            if tool_name == "query_reminder":
                reminders = payload.get("reminders", [])
                summary = "；".join([f"{r.get('time_text','')} {r.get('task','')}" for r in reminders[:3]])
                return f"我帮您查到了这些提醒：{summary}。"
            if tool_name == "update_reminder":
                return "已经帮您更新这个提醒了。"
            if tool_name == "delete_reminder":
                return "已经帮您删除这个提醒了。"
            return "好的，已完成。"

        if status == "missing_fields":
            fields = payload.get("missing_fields", [])
            return f"好的，我来帮您处理。还缺少这些信息：{', '.join(fields)}。"
        if status == "ambiguous":
            return "我找到了多个匹配的提醒，您想操作哪一个？"
        if status == "not_found":
            return "我没有找到匹配的提醒，您要不要换个条件试试？"
        return "抱歉，刚才处理时出了点问题，我们再试一次。"

    def build_tool_sample(
        self,
        user_text: str,
        tool_name: str,
        args: dict[str, Any],
        setup: Callable[[ReminderService], None] | None = None,
    ) -> dict[str, Any]:
        self.reset_storage()
        if setup:
            setup(self.service)

        call_id = self._next_call_id()
        tool_call = self._tool_call_msg(tool_name=tool_name, args=args, call_id=call_id)
        tool_message = self.executor.execute_tool_call(tool_call["tool_calls"][0])
        payload = json.loads(tool_message["content"])
        assistant_final = {
            "role": "assistant",
            "content": self._assistant_reply(tool_name=tool_name, payload=payload, user_text=user_text),
        }

        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                tool_call,
                tool_message,
                assistant_final,
            ],
            "tools": copy.deepcopy(self.tools),
        }

    def build_no_tool_sample(self, user_text: str, assistant_text: str) -> dict[str, Any]:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "tools": copy.deepcopy(self.tools),
        }


def generate_samples() -> list[dict[str, Any]]:
    storage_path = Path("data/_training_tmp_reminders.json")
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    builder = DataBuilder(storage_path=storage_path)

    samples: list[dict[str, Any]] = []

    create_cases = [
        ("明天下午4点提醒我吃药", {"time_text": "明天下午4点", "task": "吃药", "target": "self"}),
        ("后天早上7点提醒我量血压", {"time_text": "后天早上7点", "task": "量血压", "target": "self"}),
        ("今晚9点提醒我关窗", {"time_text": "今晚9点", "task": "关窗", "target": "self"}),
        ("周五上午提醒我复查", {"time_text": "周五上午", "task": "复查", "target": "self"}),
        ("提醒我明天去打球", {"time_text": "明天", "task": "打球", "target": "self"}),
        ("下周一8点提醒我买菜", {"time_text": "下周一8点", "task": "买菜", "target": "self"}),
    ]
    for user_text, args in create_cases:
        samples.append(builder.build_tool_sample(user_text=user_text, tool_name="create_reminder", args=args))

    samples.append(
        builder.build_tool_sample(
            user_text="明天下午提醒我",
            tool_name="create_reminder",
            args={"time_text": "明天下午", "target": "self"},
        )
    )

    def setup_query_success(service: ReminderService) -> None:
        service.create_reminder("明天", "吃药", "self")
        service.create_reminder("明天", "打电话", "self")

    samples.append(
        builder.build_tool_sample(
            user_text="我忘了明天要干嘛，帮我查一下提醒",
            tool_name="query_reminder",
            args={"time_text": "明天", "target": "self"},
            setup=setup_query_success,
        )
    )

    samples.append(
        builder.build_tool_sample(
            user_text="帮我查查后天有什么提醒",
            tool_name="query_reminder",
            args={"time_text": "后天", "target": "self"},
            setup=None,
        )
    )

    def setup_update_success(service: ReminderService) -> None:
        service.create_reminder("明天下午4点", "打球", "self")

    samples.append(
        builder.build_tool_sample(
            user_text="把明天下午4点打球改成晚上8点",
            tool_name="update_reminder",
            args={
                "time_text": "明天下午4点",
                "task": "打球",
                "new_time_text": "明天晚上8点",
                "target": "self",
            },
            setup=setup_update_success,
        )
    )

    def setup_update_ambiguous(service: ReminderService) -> None:
        service.create_reminder("明早8点", "吃药", "self")
        service.create_reminder("今晚8点", "吃药", "self")

    samples.append(
        builder.build_tool_sample(
            user_text="把吃药提醒改到晚上9点",
            tool_name="update_reminder",
            args={"task": "吃药", "new_time_text": "晚上9点", "target": "self"},
            setup=setup_update_ambiguous,
        )
    )

    samples.append(
        builder.build_tool_sample(
            user_text="帮我改一下提醒",
            tool_name="update_reminder",
            args={"task": "吃药", "target": "self"},
            setup=setup_update_ambiguous,
        )
    )

    def setup_delete_success(service: ReminderService) -> None:
        service.create_reminder("明天下午", "买菜", "self")

    samples.append(
        builder.build_tool_sample(
            user_text="把明天下午买菜的提醒删掉",
            tool_name="delete_reminder",
            args={"time_text": "明天下午", "task": "买菜", "target": "self"},
            setup=setup_delete_success,
        )
    )

    def setup_delete_ambiguous(service: ReminderService) -> None:
        service.create_reminder("明早8点", "吃药", "self")
        service.create_reminder("今晚8点", "吃药", "self")

    samples.append(
        builder.build_tool_sample(
            user_text="把吃药提醒删掉",
            tool_name="delete_reminder",
            args={"task": "吃药", "target": "self"},
            setup=setup_delete_ambiguous,
        )
    )

    samples.append(
        builder.build_tool_sample(
            user_text="删除我的提醒",
            tool_name="delete_reminder",
            args={"target": "self"},
            setup=setup_delete_ambiguous,
        )
    )

    no_tool_cases = [
        ("今天下午女儿来看我了，我挺开心的。", "听起来真好，和家人相聚的时光很温暖。"),
        ("我最近老是睡不好。", "辛苦了，您愿意和我说说最近作息有什么变化吗？"),
        ("你叫什么名字？", "我是小暖同学，很高兴陪您聊天。"),
        ("今天有点想老伴了。", "我在这儿陪您，想念是很自然也很深情的感受。"),
        ("你会讲个笑话吗？", "当然可以：医生问哪里不舒服，病人说一想到上班就不舒服。"),
        ("外面下雨了。", "下雨天注意路滑，也可以泡杯热茶暖暖身子。"),
        ("我孙子考试结束了。", "那真好，等他回来可以好好庆祝一下。"),
        ("今天晚饭吃什么好？", "可以试试清淡一点，比如青菜豆腐和一碗汤。"),
        ("最近记性不太好。", "很多人都会有这种时候，我们可以慢慢找适合您的办法。"),
        ("你觉得散步有用吗？", "有帮助的，规律散步对睡眠和情绪都常有好处。"),
    ]
    for user_text, assistant_text in no_tool_cases:
        samples.append(builder.build_no_tool_sample(user_text=user_text, assistant_text=assistant_text))

    if storage_path.exists():
        storage_path.unlink()

    return samples


def export_jsonl(output_path: Path) -> int:
    samples = generate_samples()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return len(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reminder tool-use training data in JSONL format.")
    parser.add_argument("--output", type=Path, default=Path("data/training_data.jsonl"), help="Output jsonl path")
    args = parser.parse_args()

    count = export_jsonl(args.output)
    print(f"Exported {count} samples to {args.output}")


if __name__ == "__main__":
    main()
