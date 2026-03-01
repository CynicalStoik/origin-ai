import uuid
from config import embed, SUMMARY_UPDATE_EVERY_N_TURNS, BOOST_AMOUNT
from memory.stm import ShortTermMemory
from memory.episodic import EpisodicMemory
from memory.facts import FactMemory
from memory.compact_memory import CompactMemory
from memory.router import route_memory
from memory.reference_detector import detect_rereference
from memory.extractor import extract_memory, update_summary, update_persona


class MemoryManager:
    def __init__(self):
        self.stm = ShortTermMemory()
        self.episodic = EpisodicMemory(embed)
        self.facts = FactMemory(embed)
        self.compact = CompactMemory()
        self.turn_count = 0

    def before_response(self, user_message: str) -> dict:
        is_reref, hint = detect_rereference(user_message)
        if is_reref and hint:
            self.episodic.boost_memory(hint, boost_amount=BOOST_AMOUNT)
            self.facts.boost_memory(hint, boost_amount=BOOST_AMOUNT)

        layers = route_memory(user_message)

        best_episode = None
        best_fact = None
        compact = None

        if "episodic" in layers:
            best_episode = self.episodic.retrieve_best(user_message)

        if "facts" in layers:
            best_fact = self.facts.retrieve_best(user_message)

        if "compact" in layers:
            compact = self.compact.get()

        return {
            "stm": self.stm.get(),
            "best_episode": best_episode,
            "best_fact": best_fact,
            "compact": compact,
            "layers_used": layers,
            "is_rereference": is_reref,
            "rereference_hint": hint
        }

    def after_response(self, user_msg: str, agent_reply: str):
        self.stm.add("user", user_msg)
        self.stm.add("assistant", agent_reply)
        self.turn_count += 1

        extracted = extract_memory(user_msg, agent_reply)

        if extracted["episode"]:
            self.episodic.add(
                text=extracted["episode"],
                tags=extracted["episode_tags"],
                episode_id=str(uuid.uuid4())
            )

        for fact in extracted["facts"]:
            self.facts.add(
                content=fact["content"],
                tags=fact.get("tags", []),
                fact_type=fact.get("type", "semantic"),
                time_span=fact.get("time_span", "ongoing")
            )

        if self.turn_count % SUMMARY_UPDATE_EVERY_N_TURNS == 0:
            self._refresh_compact()

    def _refresh_compact(self):
        recent_turns = self.stm.get()
        previous_summary = self.compact.get_summary()

        new_summary = update_summary(
            previous_summary=previous_summary,
            recent_turns=recent_turns,
            n=SUMMARY_UPDATE_EVERY_N_TURNS
        )
        self.compact.update_summary(new_summary)

        best_episode = self.episodic.retrieve_best(new_summary) if new_summary else None
        best_fact = self.facts.retrieve_best(new_summary) if new_summary else None

        updated_persona = update_persona(
            self.compact.get_persona(),
            best_episode,
            best_fact
        )
        self.compact.update_persona(updated_persona)