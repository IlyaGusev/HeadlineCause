# News entailment and causation

#### The PASCAL Recognising Textual Entailment Challenge
- 2005
- https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.295.4483&rep=rep1&type=pdf
- 1696 GS citations
- Статья по классическому RTE. Содержит в том числе опредление text entailment.

#### Generating an Entailment Corpus from News Headlines
- 2005
- https://aclanthology.org/W05-1209.pdf
- 41 GS citations
- Entailment корпус из первых параграфов и заголовков.

#### Text Mining for Causal Relations
- 2002
- https://www.cs.toronto.edu/~frank/csc2501/Readings/A4/Girju.2002.Text.pdf
- 234 GS citations
- Добыча causal relations из текстов шаблонами

#### Learning Causality for News Events Prediction
- 2012
- http://mail.kiraradinsky.com/files/Radinsky-Causality.pdf
- 188 CS citations
- Предсказание causation с помощью правил, построение графов знаний и causality графов, предсказания исходов события,, валидация толокерами

#### Learning to predict from textual data
- 2013
- https://www.jair.org/index.php/jair/article/download/10792/25765
- 49 CS citations
- Продолжение статьи Learning Causality for News Events Prediction, есть красивая схема для неё

#### Extracting Causal Relations between News Topics from Distributed Sources
- 2013
- https://core.ac.uk/download/pdf/236369528.pdf
- 2 CS citations
- Глава про causal relations, вытаскивание из Reuters предложений с маркерами causality

#### Emergent: a novel data-set for stance classification
- 2016
- https://aclanthology.org/N16-1138.pdf
- 269 CS citations
- Совсем другая постановка задачи: оцениваем заголовки против claim'а: for, against или observing

#### News Headline Grouping as a Challenging NLU Task:
- 2021
- https://arxiv.org/abs/2105.05391
- 0 CS citations
- Про кластеризацию заголовков

#### Constructing and Embedding Abstract Event Causality Networks from Text Snippets
- 2017
- http://ir.hit.edu.cn/~sdzhao/CausalEmbedding.pdf
- 53 CS citations
- Майнят из текстов причинные связки "... because of ...", учат на них что-то типа word2vec, получают пространство эмбеддингов про каузальность

#### News Event Prediction using Causality Approach on South China Sea Conflict
- 2021
- https://ieeexplore.ieee.org/document/9392431
- 0 CS citations
- Майнят из новостей связки событий в военном конфликте, учатся на них предсказывать следующие события

#### Assessing Causality Structures learned from Digital Text Media
- 2020
- https://www.researchgate.net/profile/Fernando-Delbianco/publication/345678215_Assessing_Causality_Structures_learned_from_Digital_Text_Media/links/5faa98ab299bf15bae063628/Assessing-Causality-Structures-learned-from-Digital-Text-Media.pdf
- 0 CS citations
- На разметке учат RNN выделять из потока новостей события, потом кластеризуют события по BERT-эмбеддингам, потом применяют всякие статистические тесты на каузальность (типа  Granger causality), чтобы понять, какие паттерны какой тест вытаскивает.

