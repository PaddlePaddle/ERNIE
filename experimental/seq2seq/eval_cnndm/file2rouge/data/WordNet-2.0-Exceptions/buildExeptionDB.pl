#!/usr/bin/perl -w
use DB_File;
@ARGV!=3&&die "Usage: buildExceptionDB.pl WordNet-exception-file-directory exception-file-extension output-file\n";
opendir(DIR,$ARGV[0])||die "Cannot open directory $ARGV[0]\n";
tie %exceptiondb,'DB_File',"$ARGV[2]",O_CREAT|O_RDWR,0640,$DB_HASH or
    die "Cannot open exception db file for output: $ARGV[2]\n";
while(defined($file=readdir(DIR))) {
    if($file=~/\.$ARGV[1]$/o) {
	print $file,"\n";
	open(IN,"$file")||die "Cannot open exception file: $file\n";
	while(defined($line=<IN>)) {
	    chomp($line);
	    @tmp=split(/\s+/,$line);
	    $exceptiondb{$tmp[0]}=$tmp[1];
	    print $tmp[0],"\n";
	}
	close(IN);
    }
}
untie %exceptiondb;

