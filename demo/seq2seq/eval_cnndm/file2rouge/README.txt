A Brief Introduction of the ROUGE Summary Evaluation Package
by Chin-Yew LIN 
Univeristy of Southern California/Information Sciences Institute
05/26/2005

<<WHAT'S NEW>>

(1) Correct the resampling routine which ignores the last evaluation
    item in the evaluation list. Therefore, the average scores reported
    by ROUGE is only based on the first N-1 evaluation items.
    Thanks Barry Schiffman at Columbia University to report this bug.
    This bug only affects ROUGE-1.5.X. For pre-1.5 ROUGE, it only affects
    the computation of confidence interval (CI) estimation, i.e. CI is only
    estimated by the first N-1 evaluation items, but it *does not* affect
    average scores.
(2) Correct stemming on multi-token BE heads and modifiers.
    Previously, only single token heads and modifiers were assumed.
(3) Change read_text and read_text_LCS functions to read exact words or
    bytes required by users. Previous versions carry out whitespace 
    compression and other string clear up actions before enforce the length
    limit. 
(4) Add the capability to score summaries in Basic Element (BE)
    format by using option "-3", standing for BE triple. There are 6
    different modes in BE scoring. We suggest using *"-3 HMR"* on BEs
    extracted from Minipar parse trees based on our correlation analysis
    of BE-based scoring vs. human judgements on DUC 2002 & 2003 automatic
    summaries.
(5) ROUGE now generates three scores (recall, precision and F-measure)
    for each evaluation. Previously, only one score is generated
    (recall). Precision and F-measure scores are useful when the target
    summary length is not enforced. Only recall scores were necessary since 
    DUC guideline dictated the limit on summary length. For comparison to
    previous DUC results, please use the recall scores. The default alpha
    weighting for computing F-measure is 0.5. Users can specify a
    particular alpha weighting that fits their application scenario using
    option "-p alpha-weight". Where *alpha-weight* is a number between 0
    and 1 inclusively.
(6) Pre-1.5 version of ROUGE used model average to compute the overall
    ROUGE scores when there are multiple references. Starting from v1.5+,
    ROUGE provides an option to use the best matching score among the
    references as the final score. The model average option is specified
    using "-f A" (for Average) and the best model option is specified
    using "-f B" (for the Best). The "-f A" option is better when use
    ROUGE in summarization evaluations; while "-f B" option is better when
    use ROUGE in machine translation (MT) and definition
    question-answering (DQA) evaluations since in a typical MT or DQA
    evaluation scenario matching a single reference translation or
    definition answer is sufficient. However, it is very likely that
    multiple different but equally good summaries exist in summarization
    evaluation.
(7) ROUGE v1.5+ also provides the option to specify whether model unit
    level average will be used (macro-average, i.e. treating every model
    unit equally) or token level average will be used (micro-average,
    i.e. treating every token equally). In summarization evaluation, we
    suggest using model unit level average and this is the default setting
    in ROUGE. To specify other average mode, use "-t 0" (default) for
    model unit level average, "-t 1" for token level average and "-t 2"
    for output raw token counts in models, peers, and matches.
(8) ROUGE now offers the option to use file list as the configuration
    file. The input format of the summary files are specified using the
    "-z INPUT-FORMAT" option. The INPUT-FORMAT can be SEE, SPL, ISI or
    SIMPLE. When "-z" is specified, ROUGE assumed that the ROUGE
    evaluation configuration file is a file list with each evaluation
    instance per line in the following format:

peer_path1 model_path1 model_path2 ... model_pathN
peer_path2 model_path1 model_path2 ... model_pathN
...
peer_pathM model_path1 model_path2 ... model_pathN

  The first file path is the peer summary (system summary) and it
  follows with a list of model summaries (reference summaries) separated
  by white spaces (spaces or tabs).
(9) When stemming is applied, a new WordNet exception database based
    on WordNet 2.0 is used. The new database is included in the data
    directory.

<<USAGE>>

(1) Use "-h" option to see a list of options.
    Summary:
Usage: ROUGE-1.5.4.pl
         [-a (evaluate all systems)] 
         [-c cf]
         [-d (print per evaluation scores)] 
         [-e ROUGE_EVAL_HOME] 
         [-h (usage)] 
         [-b n-bytes|-l n-words] 
         [-m (use Porter stemmer)] 
         [-n max-ngram] 
         [-s (remove stopwords)] 
         [-r number-of-samples (for resampling)] 
         [-2 max-gap-length (if < 0 then no gap length limit)] 
         [-3 <H|HM|HMR|HM1|HMR1|HMR2>] 
         [-u (include unigram in skip-bigram) default no)] 
         [-U (same as -u but also compute regular skip-bigram)] 
         [-w weight (weighting factor for WLCS)] 
         [-v (verbose)] 
         [-x (do not calculate ROUGE-L)] 
         [-f A|B (scoring formula)] 
         [-p alpha (0 <= alpha <=1)] 
         [-t 0|1|2 (count by token instead of sentence)] 
         [-z <SEE|SPL|ISI|SIMPLE>] 
         <ROUGE-eval-config-file> [<systemID>]

  ROUGE-eval-config-file: Specify the evaluation setup. Three files come with the ROUGE 
            evaluation package, i.e. ROUGE-test.xml, verify.xml, and verify-spl.xml are 
            good examples.

  systemID: Specify which system in the ROUGE-eval-config-file to perform the evaluation.
            If '-a' option is used, then all systems are evaluated and users do not need to
            provide this argument.

  Default:
    When running ROUGE without supplying any options (except -a), the following defaults are used:
    (1) ROUGE-L is computed;
    (2) 95% confidence interval;
    (3) No stemming;
    (4) Stopwords are inlcuded in the calculations;
    (5) ROUGE looks for its data directory first through the ROUGE_EVAL_HOME environment variable. If
        it is not set, the current directory is used.
    (6) Use model average scoring formula.
    (7) Assign equal importance of ROUGE recall and precision in computing ROUGE f-measure, i.e. alpha=0.5.
    (8) Compute average ROUGE by averaging sentence (unit) ROUGE scores.
  Options:
    -2: Compute skip bigram (ROGUE-S) co-occurrence, also specify the maximum gap length between two words (skip-bigram)
    -u: Compute skip bigram as -2 but include unigram, i.e. treat unigram as "start-sentence-symbol unigram"; -2 has to be specified.
    -3: Compute BE score.
        H    -> head only scoring (does not applied to Minipar-based BEs).
        HM   -> head and modifier pair scoring.
        HMR  -> head, modifier and relation triple scoring.
        HM1  -> H and HM scoring (same as HM for Minipar-based BEs).
        HMR1 -> HM and HMR scoring (same as HMR for Minipar-based BEs).
        HMR2 -> H, HM and HMR scoring (same as HMR for Minipar-based BEs).
    -a: Evaluate all systems specified in the ROUGE-eval-config-file.
    -c: Specify CF\% (0 <= CF <= 100) confidence interval to compute. The default is 95\% (i.e. CF=95).
    -d: Print per evaluation average score for each system.
    -e: Specify ROUGE_EVAL_HOME directory where the ROUGE data files can be found.
        This will overwrite the ROUGE_EVAL_HOME specified in the environment variable.
    -f: Select scoring formula: 'A' => model average; 'B' => best model
    -h: Print usage information.
    -b: Only use the first n bytes in the system/peer summary for the evaluation.
    -l: Only use the first n words in the system/peer summary for the evaluation.
    -m: Stem both model and system summaries using Porter stemmer before computing various statistics.
    -n: Compute ROUGE-N up to max-ngram length will be computed.
    -p: Relative importance of recall and precision ROUGE scores. Alpha -> 1 favors precision, Alpha -> 0 favors recall.
    -s: Remove stopwords in model and system summaries before computing various statistics.
    -t: Compute average ROUGE by averaging over the whole test corpus instead of sentences (units).
        0: use sentence as counting unit, 1: use token as couting unit, 2: same as 1 but output raw counts
        instead of precision, recall, and f-measure scores. 2 is useful when computation of the final,
        precision, recall, and f-measure scores will be conducted later.
    -r: Specify the number of sampling point in bootstrap resampling (default is 1000).
        Smaller number will speed up the evaluation but less reliable confidence interval.
    -w: Compute ROUGE-W that gives consecutive matches of length L in an LCS a weight of 'L^weight' instead of just 'L' as in LCS.
        Typically this is set to 1.2 or other number greater than 1.
    -v: Print debugging information for diagnositic purpose.
    -x: Do not calculate ROUGE-L.
    -z: ROUGE-eval-config-file is a list of peer-model pair per line in the specified format (SEE|SPL|ISI|SIMPLE).

(2) Please read RELEASE-NOTE.txt for information about updates from previous versions.

(3) The following files coming with this package in the "sample-output"
    directory are the expected output of the evaluation files in the
    "sample-test" directory.
    (a) use "data" as ROUGE_EVAL_HOME, compute 95% confidence interval,
	compute ROUGE-L (longest common subsequence, default),
        compute ROUGE-S* (skip bigram) without gap length limit,
        compute also ROUGE-SU* (skip bigram with unigram),
        run resampling 1000 times,
        compute ROUGE-N (N=1 to 4),
        compute ROUGE-W (weight = 1.2), and
	compute these ROUGE scores for all systems:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-a.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -a ROUGE-test.xml

    (b) Same as (a) but apply Porter's stemmer on the input:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-a-m.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -m -a ROUGE-test.xml

    (c) Same as (b) but apply also a stopword list on the input:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-a-m-s.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -m -s -a ROUGE-test.xml

    (d) Same as (a) but apply a summary length limit of 10 words:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-l10-a.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -l 10 -a ROUGE-test.xml

    (e) Same as (d) but apply Porter's stemmer on the input:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-l10-a-m.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -l 10 -m -a ROUGE-test.xml

    (f) Same as (e) but apply also a stopword list on the input:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-l10-a-m-s.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -l 10 -m -s -a ROUGE-test.xml

    (g) Same as (a) but apply a summary lenght limit of 75 bytes:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-b75-a.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -b 75 -a ROUGE-test.xml

    (h) Same as (g) but apply Porter's stemmer on the input:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-b75-a-m.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -b 75 -m -a ROUGE-test.xml

    (i) Same as (h) but apply also a stopword list on the input:
    ROUGE-test-c95-2-1-U-r1000-n4-w1.2-b75-a-m-s.out        
    > ROUGE-1.5.4.pl -e data -c 95 -2 -1 -U -r 1000 -n 4 -w 1.2 -b 75 -m -s -a ROUGE-test.xml

  Sample DUC2002 data (1 system and 1 model only per DUC 2002 topic), their BE and
    ROUGE evaluation configuration file in XML and file list format,
    and their expected output are also included for your reference.

    (a) Use DUC2002-BE-F.in.26.lst, a BE files list, as ROUGE the
        configuration file:
        command> ROUGE-1.5.4.pl -3 HM -z SIMPLE DUC2002-BE-F.in.26.lst 26
	output:  DUC2002-BE-F.in.26.lst.out
    (b) Use DUC2002-BE-F.in.26.simple.xml as ROUGE XML evaluation configuration file:
        command> ROUGE-1.5.4.pl -3 HM DUC2002-BE-F.in.26.simple.xml 26
	output:  DUC2002-BE-F.in.26.simple.out
    (c) Use DUC2002-BE-L.in.26.lst, a BE files list, as ROUGE the
        configuration file:
        command> ROUGE-1.5.4.pl -3 HM -z SIMPLE DUC2002-BE-L.in.26.lst 26
	output:  DUC2002-BE-L.in.26.lst.out
    (d) Use DUC2002-BE-L.in.26.simple.xml as ROUGE XML evaluation configuration file:
        command> ROUGE-1.5.4.pl -3 HM DUC2002-BE-L.in.26.simple.xml 26
	output:  DUC2002-BE-L.in.26.simple.out
    (e) Use DUC2002-ROUGE.in.26.spl.lst, a BE files list, as ROUGE the
        configuration file:
        command> ROUGE-1.5.4.pl -n 4 -z SPL DUC2002-ROUGE.in.26.spl.lst 26
	output:  DUC2002-ROUGE.in.26.spl.lst.out
    (f) Use DUC2002-ROUGE.in.26.spl.xml as ROUGE XML evaluation configuration file:
        command> ROUGE-1.5.4.pl -n 4 DUC2002-ROUGE.in.26.spl.xml 26
	output:  DUC2002-ROUGE.in.26.spl.out

<<INSTALLATION>>

(1) You need to have DB_File installed. If the Perl script complains
    about database version incompatibility, you can create a new
    WordNet-2.0.exc.db by running the buildExceptionDB.pl script in
    the "data/WordNet-2.0-Exceptions" subdirectory.
(2) You also need to install XML::DOM from http://www.cpan.org.
    Direct link: http://www.cpan.org/modules/by-module/XML/XML-DOM-1.43.tar.gz.
    You might need install extra Perl modules that are required by
    XML::DOM.
(3) Setup an environment variable ROUGE_EVAL_HOME that points to the
    "data" subdirectory. For example, if your "data" subdirectory
    located at "/usr/local/ROUGE-1.5.4/data" then you can setup
    the ROUGE_EVAL_HOME as follows:
    (a) Using csh or tcsh:
        $command_prompt>setenv ROUGE_EVAL_HOME /usr/local/ROUGE-1.5.4/data
    (b) Using bash
        $command_prompt>ROUGE_EVAL_HOME=/usr/local/ROUGE-1.5.4/data
	$command_prompt>export ROUGE_EVAL_HOME
(4) Run ROUGE-1.5.4.pl without supplying any arguments will give
    you a description of how to use the ROUGE script.
(5) Please look into the included ROUGE-test.xml, verify.xml. and
    verify-spl.xml evaluation configuration files for preparing your
    own evaluation setup. More detailed description will be provided
    later. ROUGE-test.xml and verify.xml specify the input from
    systems and references are in SEE (Summary Evaluation Environment)
    format (http://www.isi.edu/~cyl/SEE); while verify-spl.xml specify
    inputs are in sentence per line format.

<<DOCUMENTATION>>

(1) Please look into the "docs" directory for more information about
    ROUGE.
(2) ROUGE-Note-v1.4.2.pdf explains how ROUGE works. It was published in
    Proceedings of the Workshop on Text Summarization Branches Out
    (WAS 2004), Bacelona, Spain, 2004.
(3) NAACL2003.pdf presents the initial idea of applying n-gram
    co-occurrence statistics in automatic evaluation of
    summarization. It was publised in Proceedsings of 2003 Language
    Technology Conference (HLT-NAACL 2003), Edmonton, Canada, 2003.
(4) NTCIR2004.pdf discusses the effect of sample size on the
    reliability of automatic evaluation results using data in the past
    Document Understanding Conference (DUC) as examples. It was
    published in Proceedings of the 4th NTCIR Meeting, Tokyo, Japan, 2004.
(5) ACL2004.pdf shows how ROUGE can be applied on automatic evaluation
    of machine translation. It was published in Proceedings of the 42nd
    Annual Meeting of the Association for Computational Linguistics
    (ACL 2004), Barcelona, Spain, 2004.
(6) COLING2004.pdf proposes a new meta-evaluation framework, ORANGE, for
    automatic evaluation of automatic evaluation methods. We showed
    that ROUGE-S and ROUGE-L were significantly better than BLEU,
    NIST, WER, and PER automatic MT evalaution methods under the
    ORANGE framework. It was published in Proceedings of the 20th
    International Conference on Computational Linguistics (COLING 2004),
    Geneva, Switzerland, 2004.
(7) For information about BE, please go to http://www.isi.edu/~cyl/BE.

<<NOTE>>

    Thanks for using the ROUGE evaluation package. If you have any
questions or comments, please send them to cyl@isi.edu. I will do my
best to answer your questions.
