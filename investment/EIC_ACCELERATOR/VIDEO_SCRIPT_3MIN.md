# EIC short-proposal video: 3-minute script (draft, 7 October 2026)

The guide says to keep it simple: show the team and the motivation. Up to three people; a natural-person applicant may
include future team members or present the plan to build the team. Any EU language, with subtitles if needed.

**0:00–0:25 · Founder, to camera.**
"I'm Tero Keski-Valkama. I've spent my career building machine learning in industry, as lead ML engineer at HERE and
on multimodal foundation models, and I'm named inventor on 35 published patent documents. For years I've watched the
same problem: the data that runs factories and networks arrives as events, irregular and interleaved, and today's AI
isn't built for that."

**0:25–1:05 · The idea (screen: one event stream, interleaved processes, a race of clocks).**
"So we built a different kind of neural model. Its memories change with the real time between events. Incoming events
are routed by a race between learned clocks, and that race is exactly how event data is generated. The model knows
when something *should* have happened and didn't. Its cost per event stays constant, on ordinary CPUs."

**1:05–1:45 · Proof (screen: the leaderboard table).**
"We tested it where others publish. On the public EasyTPP event-stream benchmarks we beat the published state of the
art on three of five datasets, two of them at a fraction of the leading model's compute. On ICU sepsis prediction,
the same core beats the best published model on the official splits."

**1:45–2:25 · The product and why now (founder, or the future commercial lead).**
"Our first product detects faults early in operational event logs: assembly lines, industrial IoT, IT operations. We
are piloting it with [partner]. Customers measure us on detection lead time and false alarms, and they run us on the
CPUs they already have."

**2:25–3:00 · Team and ask (founder plus [named future team member or adviser]).**
"With EIC support we take this from validated technology to deployed product in 24 months: pilots, a production
runtime, first contracts. And we build the team in Europe to do it."
