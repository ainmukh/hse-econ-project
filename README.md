# Semi-Nonparametric Binary Choice Model (Gallant, Nychka)
Модель, которая работает лучше логистической регресси (или нет, я скоро проведу эксперименты и выясню это).
Идея следующая: метод максимального правдоподобия (далее ММП) работает плохо, если прогадать с распределением, например. Полупараметрический метод позволит получать состоятельные оценки. Вместо подбора семейства распределений для ММП, возьмем (почти) любое распределение и добавим полиномы (идея Галланта и Нички).

## Для чего это нужно? (И, главное, кому?)
Экономистам. Да, эта модель из микроэконометрики. Она нужна для модели Хекмана. Но моя работа – написать код и выяснить, поможет ли модель решать экономические задачи.

## Чтобы было сделано?

Работа началась с реализации ММП с такой аппроксимацией. Взяла логистическое распределение, потому что у него есть производящая функция моментов, и добавила полиномы. Код, который получает и оценки параметров и рисует графики лежит <a href='https://github.com/ainmukh/hse-econ-project/blob/master/python_files/density.py'>здесь</a>.

Для реализации я пользовалась результатами чужих статей (все-все указаны в тексте работы!) и численными методами. А еще символьной математикой – SymEngine. Работает быстро, если вся наша задача – получить оценки. Медленно, если хотим использовать эту функцию внутри модели. И вот тут я пошла изучать C++.

На этом этапе работы передо мной стояла задача сравнить логистическую регрессию и регрессию, под капотом которой ММП с аппроксимацией. Во время вывода аналитической формы функции распределения (а именно она нам нужна для модели) и написании кода к версиям, оценивающим эту функцию численно, мною был написан BFGS. Потому что я не нашла оптимизаторов на C++ таких, чтобы они сходились на моих функциях. В папке <a href='https://github.com/ainmukh/hse-econ-project/tree/master/aux'>aux</a> лежит этот оптимизатор. Алгоритм линейного поиска я брала из книги (Nocedal, Wright) и часть с интерполяцией – переписанный код с фортрана, который используется в SciPy.

В конечном итоге я вывела аналитическую формулу функции распределения. Ее можно найти в папке <a href='https://github.com/ainmukh/hse-econ-project/tree/master/Math'>math</a>. Там так же лежат мои самые читабельные черновики и тех. Код написан на C++, к Pyhton подключен. Все необходимое лежит в папках <a href='https://github.com/ainmukh/hse-econ-project/tree/master/python_files'>python files</a> и <a href='https://github.com/ainmukh/hse-econ-project/tree/master/cdf_files'>CDF files</a>.

Чтобы воспользоваться кодом сейчас, нужно загрузить содержимое двух последних упомянутых папок. Запустить <a href='https://github.com/ainmukh/hse-econ-project/blob/master/python_files/connect.py'>этот</a> файл и в терминале <code>python connect.py install</code>. Эта команда сгенерирует два файла, один из них – файл <code>.so</code>, он нам и нужен для импорта. Я пробовала использовать этот файл с чужих ПК. Пока что не работает. Но можно загрузить все файлы и сгенерировать файл самостоятельно. Перед запуском убедитесь в том, что в этом файле указаны ссылки на Ваши папки с библиотеками C++.

## Что делается:
Чтобы провести эксперименты на сгенерированных данных (они сгенерированы по-особенному), я выясняю, какие оптимизаторы лучше использовать, достаточно ли быстро они будут работать, если будут сходиться вообще. Параллельно ищу способы сделать установку результатов проделанной работы как можно удобнее. А еще пишу текст, в котором на академическом языке рассказываю, что это такое и почему оно нужно.
