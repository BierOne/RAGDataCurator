from llama_index.core.base.llms.types import ChatMessage, MessageRole
from typing import Dict, List
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole, ChatResponse
from llama_index.llms.openai.utils import to_openai_message_dicts
from openai import AsyncClient
from pydantic import BaseModel
from autorag.data.qa.filter.prompt import FILTER_PROMPT

NEI_PROMPT = {
	"NEI_filter": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""The following sentence is an answer about a question. You have to decide the answer implies 'i need more information'.
If the answer implies 'i need more information', return True. If not, return False.""",
			),
		]
    }
}

NEI_phrases = {
	"en": [
		"Imformation is not enough.",
		"there is not enough information",
		"I need more information",
		"Please provide more details.",
	]
}

def not_enough_information_rule_based(row: Dict, lang: str = "en") -> bool:
	assert (
		"query" in row.keys()
	), "query column is not in the DataFrame."
	NEI_phrase = NEI_phrases[lang]
	return not any(
		phrase in s for phrase in NEI_phrase for s in row["query"]
	)