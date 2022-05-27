#include <stdio.h>

static char *Ascii[] =
	{
"nUWl ","stAArt AXv hEHdER ","stAArt AXv tEHkst ","EHnd AXv tEHkst ",
"EHnd AXv trAEnsmIHSHAXn",
"EHnkwAYr ","AEk ","bEHl ","bAEkspEYs ","tAEb ","lIHnIYfIYd ",
"vERtIHkAXl tAEb ","fAOrmfIYd ","kAErAYj rIYtERn ","SHIHft AWt ",
"SHIHft IHn ","dIHlIYt ","dIHvIHs kAAntrAAl wAHn ","dIHvIHs kAAntrAAl tUW ",
"dIHvIHs kAAntrAAl THrIY ","dIHvIHs kAAntrAAl fOWr ","nAEk ","sIHnk ",
"EHnd tEHkst blAAk ","kAEnsEHl ","EHnd AXv mEHsIHj ","sUWbstIHtUWt ",
"EHskEYp ","fAYEHld sIYpERAEtER ","grUWp sIYpERAEtER ","rIYkAOrd sIYpERAEtER ",
"yUWnIHt sIYpERAEtER ","spEYs ","EHksklAEmEYSHAXn mAArk ","dAHbl kwOWt ",
"nUWmbER sAYn ","dAAlER sAYn ","pERsEHnt ","AEmpERsAEnd ","kwOWt ",
"OWpEHn pEHrEHn ","klOWz pEHrEHn ","AEstEHrIHsk ","plAHs ","kAAmmAX ",
"mIHnAHs ","pIYrIYAAd ","slAESH ",

"zIHrOW ","wAHn ","tUW ","THrIY ","fOWr ",
"fAYv ","sIHks ","sEHvAXn ","EYt ","nAYn ",

"kAAlAXn ","sEHmIHkAAlAXn ","lEHs DHAEn ","EHkwAXl sAYn ","grEYtER DHAEn ",
"kwEHsCHAXn mAArk ","AEt sAYn ",

"EY ","bIY ","sIY ","dIY ","IY ","EHf ","jIY  ",
"EYtCH ","AY ","jEY ","kEY ","EHl ","EHm ","EHn ","AA ","pIY ",
"kw ","AAr ","EHz ","tIY ","AHw ","vIY ",
"dAHblyUWw ","EHks ","wAYIY ","zIY ",

"lEHft brAEkEHt ","bAEkslAESH ","rAYt brAEkEHt ","kAErEHt ",
"AHndERskAOr ","AEpAAstrAAfIH ",

"EY ","bIY ","sIY ","dIY ","IY ","EHf ","jIY  ",
"EYtCH ","AY ","jEY ","kEY ","EHl ","EHm ","EHn ","AA ","pIY ",
"kw ","AAr ","EHz ","tIY ","AHw ","vIY ",
"dAHblyUWw ","EHks ","wAYIY ","zIY ",

"lEHft brEYs ","vERtIHkAXl bAAr ","rAYt brEYs ","tAYld ","dEHl ",
	};

say_ascii(character)
	int character;
	{
	outstring(Ascii[character&0x7F]);
	}

spell_word(word)
	char *word;
	{
	for (word++ ; word[1] != '\0' ; word++)
		outstring(Ascii[(*word)&0x7F]);
	}
