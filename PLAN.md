# Plan

ilya-gusev:
- ~~Бейзлайн (например tf-idf)~~
- Начать накидывать статью
- ~~Сделать первую версию чистого датасета~~
- ~~Запуск разметки пар на английском (с машинным переводом на русский на всякий случай)~~
- Уверенная разметка минимум 3к русских и 3к английских пар примеров.
- ~~Обучение берта на разметке на 3 класса: следствие слева направо, следствие справа налево, нет следствия (CauseTask)~~. ~~Во-первых, отдельная модель на каждый язык. Во-вторых, одна общая модель. Отдельные модели делаем в двух вариантах: на моноязычных бертах, на мультиязычном берте.~~
- ~~Для мультиязычного смотрим качество переноса.~~
- ~~Добавить симметричные примеры в train~~


altsoph:
- ~~Обучение берта для детекции отмен среди следствий (CancelTask)~~
- ~~Unsupervised GPT для определения направления следствия (UnsupervisedDirectionTask)~~
- ~~Найти навык английских толокеров~~ https://toloka.yandex.com/requester/quality/skill/23048 / en_gramm_native
- ~~Статистика по используемым парам глаголов~~
- ~~Эксперименты с нарративами: пытаемся строить цепочки на основе CauseTask классификатора.~~
- ~~GPT генерация последствий~~
- ~~Правда ли, что там где модель ошибается, согласованность толокеров тоже ниже~~

backlog:
- Аннотирование мета-данными: оценки bert'а, оценки single-bert'а
- Single-sentence bert для английского
- Dawid-Skene: разобраться в причинах проблемы, построить дифф двух датасетов


deep backlog
- Отдельную подвыборку выделяем как SuperGLUE-style датасет (CauseTaskHard): не более 500 примеров в обучающей выборке. Приниципы выделения примеров надо обсудить, возможные варианты: наиболее далёкие друг от друга, с наименьшим перекрытием, с наибольшим перекрытием. 


### Ревью 7TG5
* Замечания:
  * Мало опровержений
  * Ad-hoc набор кандидатов
  * Нет лицензии на заголовки
  * Не уточнены sensitive topics
* Вопросы:
  * Почему исследователям стоит браться за этот датасет?
  * Разве задача уже не решена?
  * Сколько native English speakers среди разметчиков?
* По тексту:
  * section 3.8 не нужна
  * Figure 2 не нужна
  * Table 9 не нужна
  * GPT тоже выкинуть
  * overusing conjunctive adverbs (moreover, however etc.., for example in l. 15, l.16)
  * colloquial language ('[...] graphs to understand things', l.14,)
  * verbose language (eg. l. 53).
  * l.47 should be a paragraph
  * l.42-43 should not be its own paragraph
  * l.25 to 30 goes to0 much into detail already.
  * В целом, the paper does not read very fluently

### Ревью LawZ:
* Замечания:
  * Не fair use, нет авторов, есть опасения в легальности распространения такого датасета
  * Не только новостные заголовки
  * Меньше половины МРОТа для толокеров, 45 центов - слишком мало
  * Оригинальные датасеты без лицензий
  * Есть опасения, что в датасет попали только лёгкие примеры, и именно это объясняет хорошие цифры
  * Не описана процедура получения оригинальных заголовков в tg
* Вопросы:
  * В основном слишком лёгкие примеры?
  * Английский датасет аннотировали русские?
  * Есть пользователи со странными шаблонами разметки?
  * Почему F1 full left-right-causality больше F1 simple left-right?
  * Почему F1 full right-left-causality меньше F1 full left-right-causality несмотря на аугментации?
* По тексту:
  * Page 1, line 32: "mined" -> "obtained"? Please use "collect", "obtain", "analyze", "extract", etc. as appropriate.
  * Figure 4: сделать общие оси
  * "too minimal" -> "too small"

### Ревью kY3y:
* Вопросы:
  * Как разделять опровержения и причинности? (тут я сам не понял вопроса)
  * Зачем предсказывать опровержения? (а это мы вроде уже уточняли в тексте)
  * Чем наш датасет лучше COPA?
* По тексту:
  * Просто можно лучше

### Ревью Sr6B:
* Замечания:
  * На самом деле не причнность, а "X is a precondition for Y"
  * Нет доверия английской разметке
  * Нет точного описания аугментаций в тексте
  * "refute" -> "disagree"?
  * В реальности очень маленький тест, причём это сразу и не очевидно
  * Related work как раздел может и в порядке, но непонятно, как он влияет на саму работу
  * 45 центов
* Вопросы:
  * Правда ли, что использование LaBSE при сэмплинге кандидатов создаёт "an avoidable gap between results and real-world performance"?
* По тексту:
  * Заголовок, лол
  * Непонятки с Figure 2
  * irrelevant?


План с ревью:
* ~~Додать денег толокерам~~
* ~~Написать список причин, почему мы делали так как делали~~
* ~~Добавить авторов для Телеграма~~
* ~~Написать письмо Ленте~~
* Пообещать дособрать на чистых лицензиях и с нормальной платой толокерам
* Текст поправить
* Написать ревьюверам
* Упомянуть эксперимент с COPA в ответах
* Отдельная секция про этику?
* Описать процесс фильтрации неновостей в тг
