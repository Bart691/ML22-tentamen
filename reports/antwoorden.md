# Tentamen ML2022-2023

De opdracht is om de audio van 10 cijfers, uitgesproken door zowel mannen als vrouwen, te classificeren. De dataset bevat timeseries met een wisselende lengte.

In [references/documentation.html](references/documentation.html) lees je o.a. dat elke timestep 13 features heeft.
Jouw junior collega heeft een neuraal netwerk gebouwd, maar het lukt hem niet om de accuracy boven de 67% te krijgen. Aangezien jij de cursus Machine Learning bijna succesvol hebt afgerond hoopt hij dat jij een paar betere ideeen hebt.

## Vraag 1

### 1a
In `dev/scripts` vind je de file `01_model_design.py`.
Het model in deze file heeft in de eerste hidden layer 100 units, in de tweede layer 10 units, dit heeft jouw collega ergens op stack overflow gevonden en hij had gelezen dat dit een goed model zou zijn.
De dropout staat op 0.5, hij heeft in een blog gelezen dat dit de beste settings voor dropout zou zijn.

- Wat vind je van de architectuur die hij heeft uitgekozen (een Neuraal netwerk met drie Linear layers)? Wat zijn sterke en zwakke kanten van een model als dit in het algemeen? En voor dit specifieke probleem?

---
### <span style='background :yellow' > **Antwoord:**</span>
De keuze voor een relatief simpel neuraal netwerk met drie lagen ligt voor de hand om mee te starten bij een simpele dataset. Deze bevat een input layer, hidden layer en een output layer. Het sterke punt is dat dit model snel en simpel een baseline creëert en overfitting voorkomt. Het zwakke punt is dat de accuratie hoogstwaarschijnlijk te verbeteren is, mede omdat het gekozen model niet optimaal is voor deze data. Een ander model, zoals een RNN of CNN, zou geschikter zijn voor timeserie met een volgordelijkheid. 

---

- Wat vind je van de keuzes die hij heeft gemaakt in de LinearConfig voor het aantal units ten opzichte van de data? En van de dropout?

---
### <span style='background :yellow' > **Antwoord:**</span>
- Input 13 is gelijk aan het aantal attributen en klopt;
- H1 100: In de les is naar voren gekomen dat voor 10 classes, de filterinstelling rond de 100 zou moeten zijn. De keuze voor 100 lijkt dus in de goede richting te zijn. Om de stappen (filter x 2) tussen 8 t/m 512 te kunnen maken zou ik voor 128 (ipv 100) kiezen.
- H2 10: Om deze beter te laten aansluiten met H1 en de Output zou ik deze op 64 of 32 zetten. 
- Output 20: Is gelijk aan het aantal classificaties (0 t/m 9 voor man en vrouw). Indien het alleen de cijfers betreft, zonder man/vrouw classificatie dan zou dit 10 moeten zijn. 
- Dropout 0,5: Dit lijkt, voor een relatief kleine dataset, aan de hoge kant met 0,5 (= een verlies van de helft). Een acceptabele start van de dropout lijkt mij 0,2. 
---

## 1b
Als je in de forward methode van het Linear model kijkt (in `tentamen/model.py`) dan kun je zien dat het eerste dat hij doet `x.mean(dim=1)` is. 

- Wat is het effect hiervan? Welk probleem probeert hij hier op te lossen? (maw, wat gaat er fout als hij dit niet doet?)

---
### <span style='background :yellow' > **Antwoord:** </span>
Met de code bepaald men vanuit welke dimensie de mean wordt berekend, in dit geval vanuit de tweede dimensie. Door deze stap te nemen zoekt de collega aansluiting tussen de data en het simpele lineaire model. Zonder aansluiting tussen de data en het model, zal het model niet werken. 

---

- Hoe had hij dit ook kunnen oplossen?

---
### <span style='background :yellow' > **Antwoord:** </span>
Door het toepassen van een average pooling functie in het model wordt hetzelfde resultaat bereikt. 

---

- Wat zijn voor een nadelen van de verschillende manieren om deze stap te doen?

---
### <span style='background :yellow' > **Antwoord:**</span>
Average pooling geeft de gemiddelde waarde. Het nadeel is een lagere kwaliteit in contrast door het gemiddelde, terwijl het voordeel is dat average pooling in staat is de plaatjes beter te exthraheren (minder effect op de waardes). 

---

### 1c
Omdat jij de cursus Machine Learning hebt gevolgd kun jij hem uitstekend uitleggen wat een betere architectuur zou zijn.

- Beschrijf de architecturen die je kunt overwegen voor een probleem als dit. Het is voldoende als je beschrijft welke layers in welke combinaties je zou kunnen gebruiken.

---
### <span style='background :yellow' > **Antwoord:** 
Ik zou een model gebruiken geschikt voor sequentiële timeserie data, zoals text en spraak, en met het vermogen om waardes te onthouden, zoals een RNN of GRU model. Een GRU is relatief simpel ten opzichte van een LSTM en een LSTM is minder geschikt voor deze simpele dataset. Een losstand RNN model zou mogelijk problemen met het geheugen kunnen krijgen, maar de GRU variant lost dit op. In principe zijn alle genoemde modellen te gebruiken, echter lijkt in dit geval de GRU de beste optie te zijn.

Het model bevat de volgende lagen:  
- abc
- abc

---

 

- Geef vervolgens een indicatie en motivatie voor het aantal units/filters/kernelsize etc voor elke laag die je gebruikt, en hoe je omgaat met overgangen (bv van 3 naar 2 dimensies). Een indicatie is bijvoorbeeld een educated guess voor een aantal units, plus een boven en ondergrens voor het aantal units. Met een motivatie laat je zien dat jouw keuze niet een random selectie is, maar dat je 1) andere problemen hebt gezien en dit probleem daartegen kunt afzetten en 2) een besef hebt van de consquenties van het kiezen van een range.

---
### <span style='background :yellow' > **Antwoord:** </span>
- Learning rate: Op basis van de uitkomsten tussentijdse opdracht is de instelling learning rate le-3 / 0,001 het beste gebleken voor de classificatie.
- Filters hidden layers: 128 lijkt een mooie instelling om te starten. In de les is naar voren gekomen dat voor 10 classes, de filterinstelling rond de 100 zou moeten zijn. Om de stappen (filter x 2) tussen 8 t/m 512 te kunnen maken is voor 128 (ipv 100) gekozen.
- Loss function: Cross-Entropy-Loss, omdat deze passend is voor een classificatie.
- Optimizer: Adam, omdat deze optimizer wordt gezien als een van de betere optimizers en minder parameters nodig heeft. Dit is ook gebleken uit de resultaten van de tussentijdse opdracht.
- Num layers: Deze zou ik op 4 zetten zodat er voldoende lagen zijn om het model te trainen.
- Dimensies: Door de flatten optie te gebruiken wordt de twee dimensionaal data omgezet naar een eendimensionale array.
---

- Geef aan wat jij verwacht dat de meest veelbelovende architectuur is, en waarom (opnieuw, laat zien dat je niet random getallen noemt, of keuzes maakt, maar dat jij je keuze baseert op ervaring die je hebt opgedaan met andere problemen).

---
### <span style='background :yellow' > **Antwoord:**</span>
Zoals eigenlijk al aangegeven bij de beantwoording van eerdere vragen zou ik gaan voor een GRU model omdat deze geschikt is voor een timeserie classificatie en het mogelijke geheugen probleem oplost. De loss functie passend bij een classificatie is Cross-Entropy-Loss. Voor de learning rate is is gekozen voor le-3 / 0,001, mede door de resultaten uit de tussentijdse opdracht. De optimizer is adam, omdat deze als een van de betere optimizer wordt gezien. Dit is ook gebleken tijdens het experiment van de tussentijdse opdracht. De num_layers op 4 zodat er voldoende lagen zijn voor een juiste training.

---

## 1d
Implementeer jouw veelbelovende model: 

- Maak in `model.py` een nieuw nn.Module met jouw architectuur
- Maak in `settings.py` een nieuwe config voor jouw model

---
### <span style='background :yellow' > **Antwoord:** </span>
De bestanden model.py, settings.py en 01_model_design.py zijn aangepast met het GRU model en bijpassende instellingen. De input is gemarkeerd met # TOEVOEGING TBV VRAAG 1D.

---

- Train het model met enkele educated guesses van parameters. 

---
### <span style='background :yellow' > **Antwoord:** </span>
Voor de eerste run is het model getraind met de volgende parameters: 
- Input: 13.
- Hidden: 64.
- Output: 20.
- Num_layers: 4.
- Dropout: 0,2.

Voor de tweede run is alleen de hidden size gewijzigd van 64 naar 128. De reden hiervoor is de input vanuit de les, dat voor 10 classes de filters rond de 100 moeten zijn (dus 128 voor een x 2 effect).

Voor de derde run heb ik geprobeerd om de output op 10 te zetten omdat ik twijfel of de output 20 wel de juiste is (getal 0 t/m 9 voor man en vrouw (=20) -> of alleen 0 t/m 9 (=10)). Door de aanpassing van 20 naar 10 krijgt de training een foutmelding, wat impliceert dat 20 de juiste output is.

Voor de tweede poging van de derde run is de hidden size gewijzigd van 128 naar 256. De reden hiervoor is dat de accuratie in de tweede run omhoog is gegaan.

Uit nieuwsgierigheid van de dropout effecten heb ik toch een vierde run uitgevoerd waarin de instellingen van de derde run zijn gebruikt. Uitzondering is de dropout, deze is van 0,2 naar 0,5 ingesteld.

---

- Rapporteer je bevindingen. Ga hier niet te uitgebreid hypertunen (dat is vraag 2), maar rapporteer (met een afbeelding in `antwoorden/img` die je linkt naar jouw .md antwoord) voor bijvoorbeeld drie verschillende parametersets hoe de train/test loss curve verloopt.

---
### <span style='background :yellow' > **Antwoord:** </span>
<figure>
  <p align = "center">
    <img src="img/gru_uitkomsten.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 1. Uitkomsten GRU trainingen (Y as = Accuratie / X as = Aantal Epochs)</b>
    </figcaption>
  </p>
</figure>
TOELICHTING.

### <span style='background :yellow' > **Antwoord:** </span>
<figure>
  <p align = "center">
    <img src="img/gru_loss.png" style="width:100%">
    <figcaption align="center">
      <b> Fig 2. Uitkomsten GRU LOSS (Y as = Loss / X as = Aantal Epochs)</b>
    </figcaption>
  </p>
</figure>
TOELICHTING.

---

- reflecteer op deze eerste verkenning van je model. Wat valt op, wat vind je interessant, wat had je niet verwacht, welk inzicht neem je mee naar de hypertuning.

---
### <span style='background :yellow' > **Antwoord:**</span>
TOELICHTING 

---


## Vraag 2
Een andere collega heeft alvast een hypertuning opgezet in `dev/scripts/02_tune.py`.

### 2a
Implementeer de hypertuning voor jouw architectuur:
- zorg dat je model geschikt is voor hypertuning
- je mag je model nog wat aanpassen, als vraag 1d daar aanleiding toe geeft. Als je in 1d een ander model gebruikt dan hier, geef je model dan een andere naam zodat ik ze naast elkaar kan zien.
- Stel dat je je model aanpast, laat dan je code zien (bv 2e model met aanpassing).

---
### <span style='background :yellow' > **Antwoord:**</span>
Het bestand 02_tune.py is aangepast met het hypertune model en bijpassende instellingen. De input is gemarkeerd met # TOEVOEGING TBV VRAAG 2A.

---

- voeg jouw model in op de juiste plek in de `tune.py` file.

---
### <span style='background :yellow' > **Antwoord:**</span>
Model is zoals bovenstaand beschreven ingevoegd in 02_tune.py.

---

- maak een zoekruimte aan met behulp van pydantic (naar het voorbeeld van LinearSearchSpace), maar pas het aan voor jouw model.

---
### <span style='background :yellow' > **Antwoord:**</span>
De zoekruimte is aangepast van LinearSearchSpace naar GRUmodelSearchSpace, zoals opgenomen in settings.py (ihkv vraag 1D).

---

- Licht je keuzes toe: wat hypertune je, en wat niet? Waarom? En in welke ranges zoek je, en waarom? Zie ook de [docs van ray over search space](https://docs.ray.io/en/latest/tune/api_docs/search_space.html#tune-sample-docs) en voor [rondom search algoritmes](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#bohb-tune-search-bohb-tunebohb) voor meer opties en voorbeelden.

---
### <span style='background :yellow' > **Antwoord:**</span>
- Model gewijzigd naar GRUmodel
- from tentamen.odel import GRUmodel & from pathlib import Path toegevoegd
- Epochs naar 50
- Zoekruimte in config gewijzigd naar SearchSpace, passend bij een RNN.


---

### 2b
- Analyseer de resultaten van jouw hypertuning; visualiseer de parameters van jouw hypertuning en sla het resultaat van die visualisatie op in `reports/img`. Suggesties: `parallel_coordinates` kan handig zijn, maar een goed gekozen histogram of scatterplot met goede kleuren is in sommige situaties duidelijker! Denk aan x en y labels, een titel en units voor de assen.

---
### <span style='background :yellow' > **Antwoord:**</span>
TOELICHTING 

---

- reflecteer op de hypertuning. Wat werkt wel, wat werkt niet, wat vind je verrassend, wat zijn trade-offs die je ziet in de hypertuning, wat zijn afwegingen bij het kiezen van een uiteindelijke hyperparametersetting.

---
### <span style='background :yellow' > **Antwoord:**</span>
TOELICHTING 

---

Importeer de afbeeldingen in jouw antwoorden, reflecteer op je experiment, en geef een interpretatie en toelichting op wat je ziet.

### 2c
- Zorg dat jouw prijswinnende settings in een config komen te staan in `settings.py`, en train daarmee een model met een optimaal aantal epochs, daarvoor kun je `01_model_design.py` kopieren en hernoemen naar `2c_model_design.py`.

## Vraag 3
### 3a
- fork deze repository.

---
### <span style='background :yellow' > **Antwoord:**</span>
De repository is gefork en beschikbaar via https://github.com/Bart691/ML22-tentamen.

---

- Zorg voor nette code. Als je nu `make format && make lint` runt, zie je dat alles ok is. Hoewel het in sommige gevallen prima is om een ignore toe te voegen, is de bedoeling dat je zorgt dat je code zoveel als mogelijk de richtlijnen volgt van de linters.

---
### <span style='background :yellow' > **Antwoord:**</span>
Door het runnen van make format && make lint is de code gecontroleerd en waar nodig gecorrigeerd. Het betreft een aantal kleine aanpassingen zoals het verwijderen van een mixed case.

---

- We werken sinds 22 november met git, en ik heb een `git crash coruse.pdf` gedeeld in les 2. Laat zien dat je in git kunt werken, door een git repo aan te maken en jouw code daarheen te pushen. Volg de vuistregel dat je 1) vaak (ruwweg elke dertig minuten aan code) commits doet 2) kleine, logische chunks van code/files samenvoegt in een commit 3) geef duidelijke beschrijvende namen voor je commit messages

---
### <span style='background :yellow' > **Antwoord:**</span>
Ik heb vanaf de start van de opdracht gewerkt via de bovenstaande methodiek door regelmatig de code te pushen met duidelijke beschrijvingen. Dit is terug te zien via https://github.com/Bart691/ML22-tentamen.

---

- Zorg voor duidelijke illustraties; voeg labels in voor x en y as, zorg voor eenheden op de assen, een titel, en als dat niet gaat (bv omdat het uit tensorboard komt) zorg dan voor een duidelijke caption van de afbeelding waar dat wel wordt uitgelegd.
- Laat zien dat je je vragen kort en bondig kunt beantwoorden. De antwoordstrategie "ik schiet met hagel en hoop dat het goede antwoord ertussen zit" levert minder punten op dan een kort antwoord waar je de essentie weet te vangen. 
- nodig mij uit (github handle: raoulg) voor je repository. 
