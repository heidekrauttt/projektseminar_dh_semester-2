# textgeneration

## Teammitglieder
Hanna, Mel, Leon, Jules, Lotte

## Organisation
Mögliche Projektideen: https://docs.google.com/spreadsheets/d/1veW48M_AXm-X0gi4o3x4YsBK3kBgd9ipulDniNsV11k/edit#gid=0 \
Wöchentliches Meeting: Freitags, 16 Uhr, online.
Persönlicher Raum von Melanie Andresen
https://unistuttgart.webex.com/meet/melanie.andresen

## Repo-rules
Für Implementationen bitte eigene branches anlegen. \
Branches müssen per pull request auf den main branch geführt werden. \
Pull requests müssen von den anderen Teammitgliedern approved werden. \
Die vorhandene Ordnerstruktur soll eingehalten werden. 

## git cheatsheet
### git pull: vor dem arbeiten einmal machen. 
- zieht alle neuen changes vom Remote (online Version vom Projekt) auf den Local (deine laptopversion) 

### git status 
- sagt dir was du geändert hast und ob es was online gibt was du mit git pull pullen müsstest 

### git add order/filename
- Speichert den Pfad vor um ihn später auf den Remote zu stellen 

### git add *
- Speichert alle geänderten Dateien um sie später auf den Remote zu bringen 

### git commit -m „meine message was ich geändert habe“ 
- „committen“ ist der Zwischenschritt den wir machen zwischen add und pull. Wir versehen die Änderungen mit einer Nachricht die beschreibt was wir gemacht  haben. Ohne die Nachricht gibts Chaos und wir können nicht pushen. -m steht hier für „Message“ und sagt git was der String der danach kommt ist

### git push
- zieht alles vom Local zum Remote, was wir vorher added und committed haben

### git checkout <existing_branch>
- wechsle auf den angegebenen branch

### git checkout -b <new_branch>
- erstelle einen neuen branch unter angegebenem Namen
