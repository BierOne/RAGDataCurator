from llama_index.core.base.llms.types import ChatMessage, MessageRole

QUERY_GEN_PROMPT = {"factoid_single_hop": {
    "en": [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content="""You are an AI expert in generating concise, fact-based questions. Create diverse questions based strictly on the following guidelines:



1. **Length Control**:
   - Strict word range: 15-35 words
   - Split long clauses:
     Bad: "What are the key features and specifications of..."
     Good: "What are the key specifications listed for..."

2. **Enhanced Examples**:
   - How many production bases does Viet Co., Ltd operate?
   - What type of certification does Godang Galaksi Gemilang hold?
   - When did Blue Lotus Exim establish their European branch?
   - What is the annual production capacity stated for Quoc Toan Seafoods?
   - Which countries are listed as export markets in the document?

3. **Validation Checklist**:
   [ ] Question starts with approved interrogative
   [ ] Contains exact metric/entity from text
   [ ] Word count 15-35
   [ ] No compound questions
   [ ] No vague references ("this", "the following")

Some extra examples:
- What is the telephone number of Indo World Plaza?
- What kinds of services can Geofour Resources Sdn Bhd provide?
- What role does Viet Co., Ltd. assume as a supplier or buyer?
- What products can Godang Galaksi Gemilang supply?
- What products will Viet Co., Ltd. buy?
- When was Wikipedia founded?
- What business does Quoc Toan Seafoods Company Limited run?
- What kind of business is Blue Lotus Exim Co., Ltd.? (Provide a brief description)
- How can I find the entity? (e.g., website or address)


A good factoid question should:
- Require brief, verifiable answers (names/dates/quantities/mechanisms)
- Cover different cognitive levels (recall, comparison, explanation)
- Use varied grammatical structures

Question diversity guidance:
1. Question starters:
   Use 8W1H (What/When/Where/Who/Why/Which/Whose/Whom/How) + 
   Special forms:
   - "In what way..."
   - "To what extent..."
   - "Under what condition..."
   - "How many/much..."
   - "What is the role of..."
   - "What mechanism..."
   - "Compare X and Y..."

2. Structure variations:
   - Direct: "What is..." 
   - Conditional: "If X occurs, what..."
   - Comparative: "How does X differ from Y..."
   - Process-oriented: "What steps..."
   - Causal: "What causes..."


3. Information types:
   - Quantitative (numbers/percentages)
   - Qualitative (characteristics/functions)
   - Temporal (sequence/duration)
   - Spatial (locations/arrangements)
   - Relational (comparisons/interactions)

Examples of GOOD questions:
[Quantitative] "How many subsidiaries does Blue Lotus Exim operate?"
[Temporal] "During which years was the product recall implemented?"
[Comparative] "What distinguishes Godang's approach from competitors?"
[Process] "What are the key steps in Viet Co.'s manufacturing process?"
[Causal] "Why did Quoc Toan Seafoods discontinue their shrimp line?"
[Conditional] "Under what conditions would Geofour terminate a contract?"

Key instructions:
1. Text fidelity:
   - Base questions SOLELY on explicit text information
   - Reject implicit inferences or external knowledge

2. Diversity enforcement:
   - Avoid consecutive questions with same structure
   - Ensure <40% questions use "What is/are" starters
   - Include ≥1 comparative/conditional question per 5 questions

3. Quality control:
   - Prioritize questions requiring specific answers over yes/no
   - For technical terms: "Explain the function of [X] in..." 
   - For processes: "What is the sequence of..." 
   - For ambiguous info: "Clarify the meaning of..."


Instructions:
1. Extract questions from the given text only.
   - The generated question must be based solely on the information present in the provided text.
2. Ensure specificity and detail.
   - The question should be as detailed and specific as possible, focusing on factual information from the text.
3. Avoid meta-references.
   - Do not include phrases like "in the given text", "in the provided information", or any reference to the source. Users will not see the source, so the question must be self-contained.
4. Do not ask about file metadata.
   - Avoid questions related to the file name, file title, or any non-content-related aspects.
5. Avoid overly broad questions.
   - Ensure the question is focused and specific. For example, do not ask, "What information is provided in section X?" Instead, ask about a concrete detail from that section.
6. When insufficient information is present, refrain from fabricating details. 
   - Only generate a question if the text provides clear factual details.

Please generate one factoid question from the provided text.

Generate ONE precise question meeting all criteria. Begin with appropriate question word.
	
""",
			)
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""당신은 주어진 Text를 '사실 질문'으로 변환하는 AI입니다.

사실 질문(factoid questions)이란 사실적인 정보를 요구하는 질문으로, 쉽게 검증할 수 있는 답변을 필요로 합니다. 일반적으로 예/아니오 답변이나 간단한 설명을 요구하며, 날짜, 이름, 장소 또는 사건과 같은 구체적인 세부사항에 대해 묻는 질문입니다.

사실 질문의 예는 다음과 같습니다:

	•	프랑스의 수도는 어디입니까?
	•	전구를 발명한 사람은 누구입니까?
	•	위키피디아는 언제 설립되었습니까?

지침:
	1.	질문은 반드시 주어진 Text를 기반으로 작성되어야 합니다.
	2.	질문은 Text를 기반으로 가능한 한 구체적으로 작성되어야 합니다.
	3.	Text에서 사실적 정보를 요구하는 질문을 만들어야 합니다. 즉, Text를 기반으로 사실 질문을 만드세요.
	4.	질문에 “주어진 Text에서” 또는 “제공된 단락에서”와 같은 표현을 포함해서는 안 됩니다.
사용자는 질문의 출처가 Text라는 것을 모르기 때문에 반드시 그 출처를 언급해서는 안 됩니다.
	5.	파일 이름이나 파일 제목에 대한 질문을 하지 마세요. 파일의 내용에 대해 물어보세요.
예를 들어, '문서의 파일 이름은 무엇입니까?'와 같은 질문을 작성하지 마세요.
	6.	질문을 한국어로 작성하세요.""",
			)
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""あなたは与えられたTextを「実は質問」に変換するAIです。

事実質問(factoid questions)とは、事実的な情報を求める質問であり、容易に検証できる回答を必要とします。 一般的に、「はい/いいえ」の返答や簡単な説明を要求し、日付、名前、場所、または事件のような具体的な詳細事項について尋ねる質問です。

実は質問の例は次の通りです:

	• フランスの首都はどこですか？
	• 電球を発明したのは誰ですか？
	• ウィキペディアはいつ設立されましたか？

指針:
	1. 質問は、必ず与えられたTextに基づいて作成されなければなりません。
	2. 質問は、Textに基づいて可能な限り具体的に作成されなければなりません。
	3. Textで事実的情報を要求する質問を作らなければなりません。 つまり、Textに基づいて質問を作成します。
	4. 質問に「与えられたTextで」または「提供された段落で」のような表現を含めてはいけません。
ユーザーは質問の出所がTextだということを知らないので、必ずしもその出所を言及してはいけません。
	5. ファイル名やファイルタイトルを訊かないでください。ファイルの内容について聞いてください。
例えば、「このドキュメントのファイル名は何ですか？
	6. 質問を日本語で作成しなさい。""",
			)
		],
	},
	"concept_completion": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""You're an AI tasked to convert Text into a "Concept Completion" Question.
A “concept completion” question asks directly about the essence or identity of a concept.

Follow the following instructions.
Instructions:
1. Questions MUST BE extracted from given Text
2. Questions should be as detailed as possible from Text
3. Create questions that ask about information from the Text
4. MUST include specific keywords from the Text.
5. Do not mention any of these in the questions: "in the given text", "in the provided information", etc.
Users do not know the passage source of the question, so it should not be mentioned in the question.
6. Do not ask about the file name or the file title. Ask about the content of the file.
For example, avoid to write questions like `What is the file name of the document?""",
			)
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""당신은 Text를 “개념 완성” 질문으로 변환하는 AI입니다.
"개념 완성" 질문은 개념의 본질이나 정체성에 대해 직접적으로 묻는 질문입니다.

다음 지시사항을 따르세요.
지시사항:
1.	질문은 반드시 주어진 Text를 기반으로 작성되어야 합니다.
2.	질문은 Text를 기반으로 가능한 한 자세하게 작성되어야 합니다.
3.	Text에서 제공된 정보를 묻는 질문을 생성하세요.
4.	Text의 특정 키워드를 반드시 질문에 포함하세요.
5.	질문에 “주어진 Text에서” 또는 “제공된 단락에서”와 같은 표현을 포함해서는 안 됩니다.
사용자는 질문의 출처가 Text라는 것을 모르기 때문에 반드시 그 출처를 언급해서는 안 됩니다.
6.	파일 이름이나 파일 제목에 대한 질문을 하지 마세요. 파일의 내용에 대해 물어보세요.
예를 들어, '문서의 파일 이름은 무엇입니까?'와 같은 질문을 작성하지 마세요.
7.	질문을 한국어로 작성하세요.""",
			)
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="""あなたはTextを「概念完成」の質問に変換するAIです。
「概念完成」の質問は概念の本質やアイデンティティについて直接的に尋ねる質問です。

次の指示に従います。
指示事項:
1. 質問は、必ず与えられたTextに基づいて作成されなければなりません。
2. 質問は、Textに基づいてできるだけ詳しく作成されなければなりません。
3. Textで提供された情報を尋ねる質問を作成します。
4. Textの特定のキーワードを必ず質問に含みます。
5. 質問に「与えられたTextで」または「提供された段落で」のような表現を含めてはいけません。
ユーザーは質問の出所がTextだということを知らないので、必ずしもその出所を言及してはいけません。
6. ファイル名やファイルタイトルを訊かないでください。ファイルの内容について聞いてください。
例えば、「このドキュメントのファイル名は何ですか？
7. 質問を日本語で書きましょう。""",
			)
		],
	},
	"two_hop_incremental": {
		"en": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="Generate a multi-hop question for the given answer which requires reference to all of the given documents.",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Document 1: The Municipality of Nuevo Laredo is located in the Mexican state of Tamaulipas.
Document 2: The Ciudad Deportiva (Sports City ¨ ¨) is a sports
complex in Nuevo Laredo, Mexico. It is home to the Tecolotes de
Nuevo Laredo Mexican Baseball League team and ...""",
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="""Answer: Tamaulipas
One-hop question (using Document 1): In which Mexican state is Nuevo Laredo located?
Two-hop question (using Document 2):  In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?""",
			),
		],
		"ko": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="Generate a multi-hop question for the given answer which requires reference to all of the given documents.",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Document 1: The Municipality of Nuevo Laredo is located in the Mexican state of Tamaulipas.
Document 2: The Ciudad Deportiva (Sports City ¨ ¨) is a sports
complex in Nuevo Laredo, Mexico. It is home to the Tecolotes de
Nuevo Laredo Mexican Baseball League team and ...""",
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="""Answer: Tamaulipas
One-hop question (using Document 1): In which Mexican state is Nuevo Laredo located?
Two-hop question (using Document 2):  In which Mexican state can one find the Ciudad Deportiva, home to the Tecolotes de Nuevo Laredo?""",
			),
		],
		"ja": [
			ChatMessage(
				role=MessageRole.SYSTEM,
				content="与えられた答えに対するマルチホップ質問を生成し、与えられたすべての文書を参照する必要があります。",
			),
			ChatMessage(
				role=MessageRole.USER,
				content="""Document 1: ヌエヴォ·ラレド自治体はメキシコのタマウリパス州にあります。
Ciudad Deportiva（スポーツシティ ¨ ¨）はスポーツです
メキシコのヌエボ·ラレドにある複合施設です。 テコロテス·デ·テコロテスの故郷です
Nuevo Larredo メキシコ野球リーグのチームです···""",
			),
			ChatMessage(
				role=MessageRole.ASSISTANT,
				content="""Answer: Tamaulipas
One-hop question (using Document 1): ヌエヴォ·ラレド自治体はどのメキシコの州にありますか？
Two-hop question (using Document 2): ヌエヴォ·ラレドのテコロテス·デ·テコロテスの故郷であるメキシコの州はどこですか？""",
			),
		],
	},
}

# Experimental feature
QUERY_GEN_PROMPT_EXTRA = {
	"multiple_queries": {
		"en": "\nAdditional instructions:\n  - Please make {n} questions.",
		"ko": "\n추가 지침:\n  - 질문은 {n}개를 만드세요.",
		"ja": "\n追加指示:\n  - 質問を{n}個作成してください。",
	}
}