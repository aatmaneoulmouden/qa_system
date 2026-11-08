# Testing Report - Aatmane Oulmouden

## Which tests came directly from the known requirements, and which were suggested by AI?

The tests that came directly from the known requirements were supported_01 through supported_05. These focus on the main purpose of the RAG system: retrieving relevant information from the documents and using it to answer factual questions. The remaining tests were suggested by AI to provide broader coverage, including paraphrased, ambiguous, unsupported, invalid, retrieval-failure, and misleading questions.

## Which AI-generated test case was most useful? Which was least useful or based on an invented assumption?

The most useful AI-generated test case was adversarial_01 because it tested whether the chatbot would follow the source documents instead of agreeing with a misleading statement. The least useful was boundary_01 because blank-input handling was based more on a general software testing assumption than on the known RAG requirements.

## Did you encounter a case where a test passed even though the chatbot answer was not actually good? Explain

Yes. One test passed because the expected keywords were present, but the answer was still incomplete and included unrelated content. For example, the Anthropic Security Engineering answer included useful information, but it focused too much on the table of contents instead of clearly explaining the actual use cases. This showed that keyword matching alone is not enough to judge answer quality.

## What were the most common failure modes: retrieval, generation, specification/oracle, or performance?

The most common failure mode was generation. The system often retrieved useful information, but some final answers included repeated text, unrelated content, or were not clearly focused on the question. Retrieval and specification/oracle issues were less common, and performance was not the main problem.

## How did latency change as workload increased?

The workload was too small to clearly show how latency changed as concurrency increased. The 0.00-second results may have been caused by timing precision or measurement issues, so the results were not enough to prove that the system scales better under a higher workload.

## What did the UAT tester notice that the automated tests did not?

The UAT tester noticed problems with answer quality that the automated tests did not fully capture. A response could contain the expected keywords and still include repeated text, unrelated content, or poor organization.

## Would you trust the same model that helps write the chatbot to be the sole generator/judge of its tests? Why or why not?

Yes, as long as the process is separated using different roles and different chats. One chat can help build the chatbot, another can generate tests, and another can evaluate the results. I would still verify important factual expectations against the source documents.

## What would you change before calling this chatbot production-ready?

Before calling it production-ready, I would improve the answer-generation step so responses are more direct, complete, and free of repeated or unrelated content. I would also improve handling for ambiguous and unsupported questions, strengthen error handling, and run larger performance and concurrency tests.
