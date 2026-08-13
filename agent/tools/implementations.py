from .registry import ToolRegistry
from agent.memory import LongTermMemory
import time
import uuid

registry = ToolRegistry()
memory = LongTermMemory()
reminders_db = {}

@registry.register(
    name="remember_fact",
    description="Save a fact into long-term memory.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "value": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["key", "value"]
    }
)
async def remember_fact(key: str, value: str, tags: list = None):
    await memory.save_fact(key, value, tags)
    return f"Fact '{key}' saved successfully."

@registry.register(
    name="recall_facts",
    description="Recall facts from long-term memory.",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        }
    }
)
async def recall_facts(query: str = None):
    return await memory.recall_facts(query)

@registry.register(
    name="forget_fact",
    description="Delete a fact from long-term memory.",
    parameters={
        "type": "object",
        "properties": {
            "key": {"type": "string"}
        },
        "required": ["key"]
    }
)
async def forget_fact(key: str):
    success = await memory.delete_fact(key)
    if success:
        return f"Fact '{key}' forgotten."
    return f"Fact '{key}' not found."

@registry.register(
    name="set_reminder",
    description="Set a reminder.",
    parameters={
        "type": "object",
        "properties": {
            "message": {"type": "string"},
            "delay_seconds": {"type": "number"}
        },
        "required": ["message", "delay_seconds"]
    }
)
async def set_reminder(message: str, delay_seconds: float) -> str:
    rid = str(uuid.uuid4())
    reminders_db[rid] = {"message": message, "time": time.time() + delay_seconds}
    
    import asyncio
    from voice_loop import VoiceLoop
    
    async def reminder_task():
        await asyncio.sleep(delay_seconds)
        if rid in reminders_db:
            if VoiceLoop._active_loop:
                await VoiceLoop._active_loop._notification_queue.put(message)
            del reminders_db[rid]
            
    asyncio.create_task(reminder_task())
    return f"Reminder set for {delay_seconds} seconds from now. ID: {rid}"

@registry.register(
    name="cancel_reminder",
    description="Cancel a reminder.",
    parameters={
        "type": "object",
        "properties": {
            "reminder_id": {"type": "string"}
        },
        "required": ["reminder_id"]
    }
)
def cancel_reminder(reminder_id: str):
    if reminder_id in reminders_db:
        del reminders_db[reminder_id]
        return "Reminder cancelled."
    return "Reminder not found."

@registry.register(
    name="list_reminders",
    description="List active reminders.",
    parameters={"type": "object", "properties": {}}
)
def list_reminders():
    now = time.time()
    active = {k: v for k, v in reminders_db.items() if v["time"] > now}
    return active

@registry.register(
    name="wiggle_tail",
    description="Wiggle the animatronic tail.",
    parameters={
        "type": "object",
        "properties": {
            "duration": {"type": "number", "description": "Duration in seconds"}
        }
    }
)
def wiggle_tail(duration: float = 1.0):
    return f"Tail wiggling for {duration} seconds."

@registry.register(
    name="flap_head",
    description="Flap the animatronic head.",
    parameters={"type": "object", "properties": {}}
)
def flap_head():
    return "Head flapped."

@registry.register(
    name="express_emotion",
    description="Express an emotion using motors.",
    parameters={
        "type": "object",
        "properties": {
            "emotion": {"type": "string", "enum": ["happy", "sad", "angry", "surprised"]}
        },
        "required": ["emotion"]
    }
)
def express_emotion(emotion: str):
    return f"Expressing emotion: {emotion}."

@registry.register(
    name="get_system_status",
    description="Get system status metrics.",
    parameters={"type": "object", "properties": {}}
)
def get_system_status():
    return {"status": "ok", "uptime": 3600}

@registry.register(
    name="emergency_stop",
    description="Stop all motors immediately.",
    parameters={"type": "object", "properties": {}}
)
def emergency_stop():
    return "All motors stopped immediately."
