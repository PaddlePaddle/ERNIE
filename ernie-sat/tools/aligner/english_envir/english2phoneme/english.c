/*
**	English to Phoneme rules.
**
**	Derived from: 
**
**	     AUTOMATIC TRANSLATION OF ENGLISH TEXT TO PHONETICS
**	            BY MEANS OF LETTER-TO-SOUND RULES
**
**			NRL Report 7948
**
**		      January 21st, 1976
**	    Naval Research Laboratory, Washington, D.C.
**
**
**	Published by the National Technical Information Service as
**	document "AD/A021 929".
**
**
**
**	The Phoneme codes:
**
**		IY	bEEt		IH	bIt
**		EY	gAte		EH	gEt
**		AE	fAt		AA	fAther
**		AO	lAWn		OW	lOne
**		UH	fUll		UW	fOOl
**		ER	mURdER		AX	About
**		AH	bUt		AY	hIde
**		AW	hOW		OY	tOY
**	
**		p	Pack		b	Back
**		t	Time		d	Dime
**		k	Coat		g	Goat
**		f	Fault		v	Vault
**		TH	eTHer		DH	eiTHer
**		s	Sue		z	Zoo
**		SH	leaSH		ZH	leiSure
**		HH	How		m	suM
**		n	suN		NG	suNG
**		l	Laugh		w	Wear
**		y	Young		r	Rate
**		CH	CHar		j	Jar
**		WH	WHere
**
**
**	Rules are made up of four parts:
**	
**		The left context.
**		The text to match.
**		The right context.
**		The phonemes to substitute for the matched text.
**
**	Procedure:
**
**		Seperate each block of letters (apostrophes included) 
**		and add a space on each side.  For each unmatched 
**		letter in the word, look through the rules where the 
**		text to match starts with the letter in the word.  If 
**		the text to match is found and the right and left 
**		context patterns also match, output the phonemes for 
**		that rule and skip to the next unmatched letter.
**
**
**	Special Context Symbols:
**
**		#	One or more vowels
**		:	Zero or more consonants
**		^	One consonant.
**		.	One of B, D, V, G, J, L, M, N, R, W or Z (voiced 
**			consonants)
**		%	One of ER, E, ES, ED, ING, ELY (a suffix)
**			(Found in right context only)
**		+	One of E, I or Y (a "front" vowel)
**
*/


/* Context definitions */
static char Anything[] = "";	/* No context requirement */
static char Nothing[] = " ";	/* Context is beginning or end of word */

/* Phoneme definitions */
static char Pause[] = " ";	/* Short silence */
static char Silent[] = "";	/* No phonemes */

#define LEFT_PART	0
#define MATCH_PART	1
#define RIGHT_PART	2
#define OUT_PART	3

typedef char *Rule[4];	/* Rule is an array of 4 character pointers */

/*0 = Punctuation */
/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule punct_rules[] =
	{
	{Anything,	" ",		Anything,	Pause	},
	{Anything,	"-",		Anything,	Silent	},
	{".",		"'S",		Anything,	"z"	},
	{"#:.E",	"'S",		Anything,	"z"	},
	{"#",		"'S",		Anything,	"z"	},
	{Anything,	"'",		Anything,	Silent	},
	{Anything,	",",		Anything,	Pause	},
	{Anything,	".",		Anything,	Pause	},
	{Anything,	"?",		Anything,	Pause	},
	{Anything,	"!",		Anything,	Pause	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule A_rules[] =
	{
	{Anything,	"A",		Nothing,	"AX"	},
	{Nothing,	"ARE",		Nothing,	"AAr"	},
	{Nothing,	"AR",		"O",		"AXr"	},
	{Anything,	"AR",		"#",		"EHr"	},
	{"^",		"AS",		"#",		"EYs"	},
	{Anything,	"A",		"WA",		"AX"	},
	{Anything,	"AW",		Anything,	"AO"	},
	{" :",		"ANY",		Anything,	"EHnIY"	},
	{Anything,	"A",		"^+#",		"EY"	},
	{"#:",		"ALLY",		Anything,	"AXlIY"	},
	{Nothing,	"AL",		"#",		"AXl"	},
	{Anything,	"AGAIN",	Anything,	"AXgEHn"},
	{"#:",		"AG",		"E",		"IHj"	},
	{Anything,	"A",		"^+:#",		"AE"	},
	{" :",		"A",		"^+ ",		"EY"	},
	{Anything,	"A",		"^%",		"EY"	},
	{Nothing,	"ARR",		Anything,	"AXr"	},
	{Anything,	"ARR",		Anything,	"AEr"	},
	{" :",		"AR",		Nothing,	"AAr"	},
	{Anything,	"AR",		Nothing,	"ER"	},
	{Anything,	"AR",		Anything,	"AAr"	},
	{Anything,	"AIR",		Anything,	"EHr"	},
	{Anything,	"AI",		Anything,	"EY"	},
	{Anything,	"AY",		Anything,	"EY"	},
	{Anything,	"AU",		Anything,	"AO"	},
	{"#:",		"AL",		Nothing,	"AXl"	},
	{"#:",		"ALS",		Nothing,	"AXlz"	},
	{Anything,	"ALK",		Anything,	"AOk"	},
	{Anything,	"AL",		"^",		"AOl"	},
	{" :",		"ABLE",		Anything,	"EYbAXl"},
	{Anything,	"ABLE",		Anything,	"AXbAXl"},
	{Anything,	"ANG",		"+",		"EYnj"	},
	{Anything,	"A",		Anything,	"AE"	},
 	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule B_rules[] =
	{
	{Nothing,	"BE",		"^#",		"bIH"	},
	{Anything,	"BEING",	Anything,	"bIYIHNG"},
	{Nothing,	"BOTH",		Nothing,	"bOWTH"	},
	{Nothing,	"BUS",		"#",		"bIHz"	},
	{Anything,	"BUIL",		Anything,	"bIHl"	},
	{Anything,	"B",		Anything,	"b"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule C_rules[] =
	{
	{Nothing,	"CH",		"^",		"k"	},
	{"^E",		"CH",		Anything,	"k"	},
	{Anything,	"CH",		Anything,	"CH"	},
	{" S",		"CI",		"#",		"sAY"	},
	{Anything,	"CI",		"A",		"SH"	},
	{Anything,	"CI",		"O",		"SH"	},
	{Anything,	"CI",		"EN",		"SH"	},
	{Anything,	"C",		"+",		"s"	},
	{Anything,	"CK",		Anything,	"k"	},
	{Anything,	"COM",		"%",		"kAHm"	},
	{Anything,	"C",		Anything,	"k"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule D_rules[] =
	{
	{"#:",		"DED",		Nothing,	"dIHd"	},
	{".E",		"D",		Nothing,	"d"	},
	{"#:^E",	"D",		Nothing,	"t"	},
	{Nothing,	"DE",		"^#",		"dIH"	},
	{Nothing,	"DO",		Nothing,	"dUW"	},
	{Nothing,	"DOES",		Anything,	"dAHz"	},
	{Nothing,	"DOING",	Anything,	"dUWIHNG"},
	{Nothing,	"DOW",		Anything,	"dAW"	},
	{Anything,	"DU",		"A",		"jUW"	},
	{Anything,	"D",		Anything,	"d"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule E_rules[] =
	{
	{"#:",		"E",		Nothing,	Silent	},
	{"':^",		"E",		Nothing,	Silent	},
	{" :",		"E",		Nothing,	"IY"	},
	{"#",		"ED",		Nothing,	"d"	},
	{"#:",		"E",		"D ",		Silent	},
	{Anything,	"EV",		"ER",		"EHv"	},
	{Anything,	"E",		"^%",		"IY"	},
	{Anything,	"ERI",		"#",		"IYrIY"	},
	{Anything,	"ERI",		Anything,	"EHrIH"	},
	{"#:",		"ER",		"#",		"ER"	},
	{Anything,	"ER",		"#",		"EHr"	},
	{Anything,	"ER",		Anything,	"ER"	},
	{Nothing,	"EVEN",		Anything,	"IYvEHn"},
	{"#:",		"E",		"W",		Silent	},
	{"T",		"EW",		Anything,	"UW"	},
	{"S",		"EW",		Anything,	"UW"	},
	{"R",		"EW",		Anything,	"UW"	},
	{"D",		"EW",		Anything,	"UW"	},
	{"L",		"EW",		Anything,	"UW"	},
	{"Z",		"EW",		Anything,	"UW"	},
	{"N",		"EW",		Anything,	"UW"	},
	{"J",		"EW",		Anything,	"UW"	},
	{"TH",		"EW",		Anything,	"UW"	},
	{"CH",		"EW",		Anything,	"UW"	},
	{"SH",		"EW",		Anything,	"UW"	},
	{Anything,	"EW",		Anything,	"yUW"	},
	{Anything,	"E",		"O",		"IY"	},
	{"#:S",		"ES",		Nothing,	"IHz"	},
	{"#:C",		"ES",		Nothing,	"IHz"	},
	{"#:G",		"ES",		Nothing,	"IHz"	},
	{"#:Z",		"ES",		Nothing,	"IHz"	},
	{"#:X",		"ES",		Nothing,	"IHz"	},
	{"#:J",		"ES",		Nothing,	"IHz"	},
	{"#:CH",	"ES",		Nothing,	"IHz"	},
	{"#:SH",	"ES",		Nothing,	"IHz"	},
	{"#:",		"E",		"S ",		Silent	},
	{"#:",		"ELY",		Nothing,	"lIY"	},
	{"#:",		"EMENT",	Anything,	"mEHnt"	},
	{Anything,	"EFUL",		Anything,	"fUHl"	},
	{Anything,	"EE",		Anything,	"IY"	},
	{Anything,	"EARN",		Anything,	"ERn"	},
	{Nothing,	"EAR",		"^",		"ER"	},
	{Anything,	"EAD",		Anything,	"EHd"	},
	{"#:",		"EA",		Nothing,	"IYAX"	},
	{Anything,	"EA",		"SU",		"EH"	},
	{Anything,	"EA",		Anything,	"IY"	},
	{Anything,	"EIGH",		Anything,	"EY"	},
	{Anything,	"EI",		Anything,	"IY"	},
	{Nothing,	"EYE",		Anything,	"AY"	},
	{Anything,	"EY",		Anything,	"IY"	},
	{Anything,	"EU",		Anything,	"yUW"	},
	{Anything,	"E",		Anything,	"EH"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule F_rules[] =
	{
	{Anything,	"FUL",		Anything,	"fUHl"	},
	{Anything,	"F",		Anything,	"f"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule G_rules[] =
	{
	{Anything,	"GIV",		Anything,	"gIHv"	},
	{Nothing,	"G",		"I^",		"g"	},
	{Anything,	"GE",		"T",		"gEH"	},
	{"SU",		"GGES",		Anything,	"gjEHs"	},
	{Anything,	"GG",		Anything,	"g"	},
	{" B#",		"G",		Anything,	"g"	},
	{Anything,	"G",		"+",		"j"	},
	{Anything,	"GREAT",	Anything,	"grEYt"	},
	{"#",		"GH",		Anything,	Silent	},
	{Anything,	"G",		Anything,	"g"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule H_rules[] =
	{
	{Nothing,	"HAV",		Anything,	"hAEv"	},
	{Nothing,	"HERE",		Anything,	"hIYr"	},
	{Nothing,	"HOUR",		Anything,	"AWER"	},
	{Anything,	"HOW",		Anything,	"hAW"	},
	{Anything,	"H",		"#",		"h"	},
	{Anything,	"H",		Anything,	Silent	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule I_rules[] =
	{
	{Nothing,	"IN",		Anything,	"IHn"	},
	{Nothing,	"I",		Nothing,	"AY"	},
	{Anything,	"IN",		"D",		"AYn"	},
	{Anything,	"IER",		Anything,	"IYER"	},
	{"#:R",		"IED",		Anything,	"IYd"	},
	{Anything,	"IED",		Nothing,	"AYd"	},
	{Anything,	"IEN",		Anything,	"IYEHn"	},
	{Anything,	"IE",		"T",		"AYEH"	},
	{" :",		"I",		"%",		"AY"	},
	{Anything,	"I",		"%",		"IY"	},
	{Anything,	"IE",		Anything,	"IY"	},
	{Anything,	"I",		"^+:#",		"IH"	},
	{Anything,	"IR",		"#",		"AYr"	},
	{Anything,	"IZ",		"%",		"AYz"	},
	{Anything,	"IS",		"%",		"AYz"	},
	{Anything,	"I",		"D%",		"AY"	},
	{"+^",		"I",		"^+",		"IH"	},
	{Anything,	"I",		"T%",		"AY"	},
	{"#:^",		"I",		"^+",		"IH"	},
	{Anything,	"I",		"^+",		"AY"	},
	{Anything,	"IR",		Anything,	"ER"	},
	{Anything,	"IGH",		Anything,	"AY"	},
	{Anything,	"ILD",		Anything,	"AYld"	},
	{Anything,	"IGN",		Nothing,	"AYn"	},
	{Anything,	"IGN",		"^",		"AYn"	},
	{Anything,	"IGN",		"%",		"AYn"	},
	{Anything,	"IQUE",		Anything,	"IYk"	},
	{Anything,	"I",		Anything,	"IH"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule J_rules[] =
	{
	{Anything,	"J",		Anything,	"j"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule K_rules[] =
	{
	{Nothing,	"K",		"N",		Silent	},
	{Anything,	"K",		Anything,	"k"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule L_rules[] =
	{
	{Anything,	"LO",		"C#",		"lOW"	},
	{"L",		"L",		Anything,	Silent	},
	{"#:^",		"L",		"%",		"AXl"	},
	{Anything,	"LEAD",		Anything,	"lIYd"	},
	{Anything,	"L",		Anything,	"l"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule M_rules[] =
	{
	{Anything,	"MOV",		Anything,	"mUWv"	},
	{Anything,	"M",		Anything,	"m"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule N_rules[] =
	{
	{"E",		"NG",		"+",		"nj"	},
	{Anything,	"NG",		"R",		"NGg"	},
	{Anything,	"NG",		"#",		"NGg"	},
	{Anything,	"NGL",		"%",		"NGgAXl"},
	{Anything,	"NG",		Anything,	"NG"	},
	{Anything,	"NK",		Anything,	"NGk"	},
	{Nothing,	"NOW",		Nothing,	"nAW"	},
	{Anything,	"N",		Anything,	"n"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule O_rules[] =
	{
	{Anything,	"OF",		Nothing,	"AXv"	},
	{Anything,	"OROUGH",	Anything,	"EROW"	},
	{"#:",		"OR",		Nothing,	"ER"	},
	{"#:",		"ORS",		Nothing,	"ERz"	},
	{Anything,	"OR",		Anything,	"AOr"	},
	{Nothing,	"ONE",		Anything,	"wAHn"	},
	{Anything,	"OW",		Anything,	"OW"	},
	{Nothing,	"OVER",		Anything,	"OWvER"	},
	{Anything,	"OV",		Anything,	"AHv"	},
	{Anything,	"O",		"^%",		"OW"	},
	{Anything,	"O",		"^EN",		"OW"	},
	{Anything,	"O",		"^I#",		"OW"	},
	{Anything,	"OL",		"D",		"OWl"	},
	{Anything,	"OUGHT",	Anything,	"AOt"	},
	{Anything,	"OUGH",		Anything,	"AHf"	},
	{Nothing,	"OU",		Anything,	"AW"	},
	{"H",		"OU",		"S#",		"AW"	},
	{Anything,	"OUS",		Anything,	"AXs"	},
	{Anything,	"OUR",		Anything,	"AOr"	},
	{Anything,	"OULD",		Anything,	"UHd"	},
	{"^",		"OU",		"^L",		"AH"	},
	{Anything,	"OUP",		Anything,	"UWp"	},
	{Anything,	"OU",		Anything,	"AW"	},
	{Anything,	"OY",		Anything,	"OY"	},
	{Anything,	"OING",		Anything,	"OWIHNG"},
	{Anything,	"OI",		Anything,	"OY"	},
	{Anything,	"OOR",		Anything,	"AOr"	},
	{Anything,	"OOK",		Anything,	"UHk"	},
	{Anything,	"OOD",		Anything,	"UHd"	},
	{Anything,	"OO",		Anything,	"UW"	},
	{Anything,	"O",		"E",		"OW"	},
	{Anything,	"O",		Nothing,	"OW"	},
	{Anything,	"OA",		Anything,	"OW"	},
	{Nothing,	"ONLY",		Anything,	"OWnlIY"},
	{Nothing,	"ONCE",		Anything,	"wAHns"	},
	{Anything,	"ON'T",		Anything,	"OWnt"	},
	{"C",		"O",		"N",		"AA"	},
	{Anything,	"O",		"NG",		"AO"	},
	{" :^",		"O",		"N",		"AH"	},
	{"I",		"ON",		Anything,	"AXn"	},
	{"#:",		"ON",		Nothing,	"AXn"	},
	{"#^",		"ON",		Anything,	"AXn"	},
	{Anything,	"O",		"ST ",		"OW"	},
	{Anything,	"OF",		"^",		"AOf"	},
	{Anything,	"OTHER",	Anything,	"AHDHER"},
	{Anything,	"OSS",		Nothing,	"AOs"	},
	{"#:^",		"OM",		Anything,	"AHm"	},
	{Anything,	"O",		Anything,	"AA"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule P_rules[] =
	{
	{Anything,	"PH",		Anything,	"f"	},
	{Anything,	"PEOP",		Anything,	"pIYp"	},
	{Anything,	"POW",		Anything,	"pAW"	},
	{Anything,	"PUT",		Nothing,	"pUHt"	},
	{Anything,	"P",		Anything,	"p"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule Q_rules[] =
	{
	{Anything,	"QUAR",		Anything,	"kwAOr"	},
	{Anything,	"QU",		Anything,	"kw"	},
	{Anything,	"Q",		Anything,	"k"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule R_rules[] =
	{
	{Nothing,	"RE",		"^#",		"rIY"	},
	{Anything,	"R",		Anything,	"r"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule S_rules[] =
	{
	{Anything,	"SH",		Anything,	"SH"	},
	{"#",		"SION",		Anything,	"ZHAXn"	},
	{Anything,	"SOME",		Anything,	"sAHm"	},
	{"#",		"SUR",		"#",		"ZHER"	},
	{Anything,	"SUR",		"#",		"SHER"	},
	{"#",		"SU",		"#",		"ZHUW"	},
	{"#",		"SSU",		"#",		"SHUW"	},
	{"#",		"SED",		Nothing,	"zd"	},
	{"#",		"S",		"#",		"z"	},
	{Anything,	"SAID",		Anything,	"sEHd"	},
	{"^",		"SION",		Anything,	"SHAXn"	},
	{Anything,	"S",		"S",		Silent	},
	{".",		"S",		Nothing,	"z"	},
	{"#:.E",	"S",		Nothing,	"z"	},
	{"#:^##",	"S",		Nothing,	"z"	},
	{"#:^#",	"S",		Nothing,	"s"	},
	{"U",		"S",		Nothing,	"s"	},
	{" :#",		"S",		Nothing,	"z"	},
	{Nothing,	"SCH",		Anything,	"sk"	},
	{Anything,	"S",		"C+",		Silent	},
	{"#",		"SM",		Anything,	"zm"	},
	{"#",		"SN",		"'",		"zAXn"	},
	{Anything,	"S",		Anything,	"s"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule T_rules[] =
	{
	{Nothing,	"THE",		Nothing,	"DHAX"	},
	{Anything,	"TO",		Nothing,	"tUW"	},
	{Anything,	"THAT",		Nothing,	"DHAEt"	},
	{Nothing,	"THIS",		Nothing,	"DHIHs"	},
	{Nothing,	"THEY",		Anything,	"DHEY"	},
	{Nothing,	"THERE",	Anything,	"DHEHr"	},
	{Anything,	"THER",		Anything,	"DHER"	},
	{Anything,	"THEIR",	Anything,	"DHEHr"	},
	{Nothing,	"THAN",		Nothing,	"DHAEn"	},
	{Nothing,	"THEM",		Nothing,	"DHEHm"	},
	{Anything,	"THESE",	Nothing,	"DHIYz"	},
	{Nothing,	"THEN",		Anything,	"DHEHn"	},
	{Anything,	"THROUGH",	Anything,	"THrUW"	},
	{Anything,	"THOSE",	Anything,	"DHOWz"	},
	{Anything,	"THOUGH",	Nothing,	"DHOW"	},
	{Nothing,	"THUS",		Anything,	"DHAHs"	},
	{Anything,	"TH",		Anything,	"TH"	},
	{"#:",		"TED",		Nothing,	"tIHd"	},
	{"S",		"TI",		"#N",		"CH"	},
	{Anything,	"TI",		"O",		"SH"	},
	{Anything,	"TI",		"A",		"SH"	},
	{Anything,	"TIEN",		Anything,	"SHAXn"	},
	{Anything,	"TUR",		"#",		"CHER"	},
	{Anything,	"TU",		"A",		"CHUW"	},
	{Nothing,	"TWO",		Anything,	"tUW"	},
	{Anything,	"T",		Anything,	"t"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule U_rules[] =
	{
	{Nothing,	"UN",		"I",		"yUWn"	},
	{Nothing,	"UN",		Anything,	"AHn"	},
	{Nothing,	"UPON",		Anything,	"AXpAOn"},
	{"T",		"UR",		"#",		"UHr"	},
	{"S",		"UR",		"#",		"UHr"	},
	{"R",		"UR",		"#",		"UHr"	},
	{"D",		"UR",		"#",		"UHr"	},
	{"L",		"UR",		"#",		"UHr"	},
	{"Z",		"UR",		"#",		"UHr"	},
	{"N",		"UR",		"#",		"UHr"	},
	{"J",		"UR",		"#",		"UHr"	},
	{"TH",		"UR",		"#",		"UHr"	},
	{"CH",		"UR",		"#",		"UHr"	},
	{"SH",		"UR",		"#",		"UHr"	},
	{Anything,	"UR",		"#",		"yUHr"	},
	{Anything,	"UR",		Anything,	"ER"	},
	{Anything,	"U",		"^ ",		"AH"	},
	{Anything,	"U",		"^^",		"AH"	},
	{Anything,	"UY",		Anything,	"AY"	},
	{" G",		"U",		"#",		Silent	},
	{"G",		"U",		"%",		Silent	},
	{"G",		"U",		"#",		"w"	},
	{"#N",		"U",		Anything,	"yUW"	},
	{"T",		"U",		Anything,	"UW"	},
	{"S",		"U",		Anything,	"UW"	},
	{"R",		"U",		Anything,	"UW"	},
	{"D",		"U",		Anything,	"UW"	},
	{"L",		"U",		Anything,	"UW"	},
	{"Z",		"U",		Anything,	"UW"	},
	{"N",		"U",		Anything,	"UW"	},
	{"J",		"U",		Anything,	"UW"	},
	{"TH",		"U",		Anything,	"UW"	},
	{"CH",		"U",		Anything,	"UW"	},
	{"SH",		"U",		Anything,	"UW"	},
	{Anything,	"U",		Anything,	"yUW"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule V_rules[] =
	{
	{Anything,	"VIEW",		Anything,	"vyUW"	},
	{Anything,	"V",		Anything,	"v"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule W_rules[] =
	{
	{Nothing,	"WERE",		Anything,	"wER"	},
	{Anything,	"WA",		"S",		"wAA"	},
	{Anything,	"WA",		"T",		"wAA"	},
	{Anything,	"WHERE",	Anything,	"WHEHr"	},
	{Anything,	"WHAT",		Anything,	"WHAAt"	},
	{Anything,	"WHOL",		Anything,	"hOWl"	},
	{Anything,	"WHO",		Anything,	"hUW"	},
	{Anything,	"WH",		Anything,	"WH"	},
	{Anything,	"WAR",		Anything,	"wAOr"	},
	{Anything,	"WOR",		"^",		"wER"	},
	{Anything,	"WR",		Anything,	"r"	},
	{Anything,	"W",		Anything,	"w"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule X_rules[] =
	{
	{Anything,	"X",		Anything,	"ks"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule Y_rules[] =
	{
	{Anything,	"YOUNG",	Anything,	"yAHNG"	},
	{Nothing,	"YOU",		Anything,	"yUW"	},
	{Nothing,	"YES",		Anything,	"yEHs"	},
	{Nothing,	"Y",		Anything,	"y"	},
	{"#:^",		"Y",		Nothing,	"IY"	},
	{"#:^",		"Y",		"I",		"IY"	},
	{" :",		"Y",		Nothing,	"AY"	},
	{" :",		"Y",		"#",		"AY"	},
	{" :",		"Y",		"^+:#",		"IH"	},
	{" :",		"Y",		"^#",		"AY"	},
	{Anything,	"Y",		Anything,	"IH"	},
	{Anything,	0,		Anything,	Silent	},
	};

/*
**	LEFT_PART	MATCH_PART	RIGHT_PART	OUT_PART
*/
static Rule Z_rules[] =
	{
	{Anything,	"Z",		Anything,	"z"	},
	{Anything,	0,		Anything,	Silent	},
	};

Rule *Rules[] =
	{
	punct_rules,
	A_rules, B_rules, C_rules, D_rules, E_rules, F_rules, G_rules, 
	H_rules, I_rules, J_rules, K_rules, L_rules, M_rules, N_rules, 
	O_rules, P_rules, Q_rules, R_rules, S_rules, T_rules, U_rules, 
	V_rules, W_rules, X_rules, Y_rules, Z_rules
	};
