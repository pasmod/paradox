#----------------------------------------------
# get the final goldstandard files
#-----------------------------------------------
use strict;
use warnings;

die "Usage:  perl  get_final.pl  sts_dir  filter_dir  gs_dir\n" if (@ARGV != 3);
#my $input_dir = "../data/raw/test";
#my $filter_dir = "../data/test3";
#my $output_dir = "../data/final";
my $input_dir = $ARGV[0];
my $filter_dir = $ARGV[1];
my $output_dir = $ARGV[2];

foreach my $filter_file (<$filter_dir/*>) {
    $filter_file =~ /$filter_dir\/(.+)/;
    my $filename = $1;
    
    ### 1. read filter file
    my %hash;
    open(I, $filter_file) || die "Cannot open $filter_file\n";
    while (my $str = <I>) {
	$str = lc($str);
	$str =~ s/\s+$//;
	my @info = split("\t", $str);
	my $k = "$info[4]\t$info[5]";
	if (!exists $hash{$k}) {
	    $hash{$k} = $info[0];
	} else {
	    die "duplicates!\n";
	}
    }
    close(I);

    ### 2. read input file
    my $input_file = "$input_dir/STS.input.$filename.txt";
    my $output_file = "$output_dir/STS.gs.$filename.txt";
    open(I, $input_file) || die "Cannot open $input_file\n";
    open(O, ">$output_file") || die "Cannot create $output_file\n";
    my $count = 0;
    my %data;
    while (my $str = <I>) {
	$str = lc($str);
	$str =~ s/\s+$//;
	if (exists $data{$str}) {
	    print O "\n";
	    next;
	} else {
	    $data{$str} = 1;
	}

	if (exists $hash{$str}) {
	    print O sprintf("%.2f", $hash{$str}) . "\n";
	    $count++;
	} else {
	    print O "\n";
	}
    }
    print "$filename\t$count\n";
    close(I);
}
