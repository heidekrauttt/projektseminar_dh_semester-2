# textgeneration

## Abstract

Im Rahmen des Semesterprojektseminars der Digital Humanities (M.A.) an der Universität Stuttgart haben wir ein Evaluationstool entworfen, das von künstlicher Intelligenz generierte Dramen aufgrund verschiedener Eigenschaften (Features) filtert. Ziel des Tools ist es, den Benutzenden eine Hilfestellung bei der qualitativen Auswahl von künstlich generierten Texten zu bieten. 
In dieser Arbeit werden wir unsere Vorgehensweise und unseren Ansatz vorstellen, um die Frage zu lösen, was einen guten Text der Überkategorie Drama ausmacht und wie man diese Qualität messbar machen kann. Außerdem werden wir ermitteln, ob das von uns erstellte Filtertool nützlich bei der Auswahl von Texten nach bestimmten Eigenschaften sein kann.
Wir werden beleuchten, welche Dramen nach einer Filterung mit den von uns empfohlenen Parametern übrig bleiben. Wir werden außerdem darstellen, in wie weit diese Dramen qualitativ herausstechen, indem wir eines der Dramen als Prompt in den Dramengenerator hineingeben und somit längere Dramen generieren. Wir konnten zeigen, dass die längeren Dramen hinsichtlich der von uns gewählten Parameter besser abschneiden als die initialen Generierungen. Nach der Filterung mit unserem Evaluationstool konnten wir in dieser verlängerten Dramengenerierung aus 100 Dramen 9 empfehlen, da sie unseren empfohlenen Wertebereichen entsprechen. Dies stellt eine Verbesserung um den Faktor 4,5 im Vergleich zur initialen Filterung (2 Empfehlungen aus 100 Dramen) dar.
Unser Ansatz zeigt, welche Chancen eine computergestützte Qualitätsanalyse von maschinell erstelltem Text bieten kann. 

## Aufbau des Repositories
Das Evaluationstool findet sich in evaluation_tool.py.
Alle Dramen finden sich im Ordner dramas, und sind nach Generation sortiert. Es gibt drei Generationen.
Generation 1 und Generation 3 enthalten je 100 generierte Dramen.
Generation 2 enthält 500 generierte Dramen.
Generation 1 und 2 erhielten als Prompt "PRINZESSIN: Zuerst Tomaten," die dritte Generation hat als Prompt eine nach der Filterung empfohlene Generierung der zweiten Generation. 

## Kontakt
Bei Fragen oder Verwendung des Codes für weitere Forschung bitten wir um vorherige Kontaktaufnahme.

## Teammitglieder
Charlotte Ammer (@heidekrauttt)
Melina Rieger (@melrieger)
Leon Rösler (@TheynT)
Julia Werner (@juliwer)
Hanna Weimann (@makina3485)