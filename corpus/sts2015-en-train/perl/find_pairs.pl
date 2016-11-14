#---------------------------------------------------
# find the pairs that are in sts2015 data set
#---------------------------------------------------

use strict;
use warnings;

die "Usage:  perl  find_pairs.pl  input_file  sts_dir  output_file\n" if (@ARGV != 3);

#my $input_file = "../data/clean/text";
#my $input_dir = "../data/raw/test";
#my $output_file = "../data/clean/text.clean";
my $input_file = $ARGV[0];
my $input_dir = $ARGV[1];
my $output_file = $ARGV[2];


### 1. read test files
my %pair_hash;
foreach my $input_file (<$input_dir/*>) {
    $input_file =~ /$input_dir\/.+\.(.+?)\.txt$/;
    my $set = $1;
    open(I, $input_file) || die "Cannot open $input_file\n";
    while (my $str = <I>) {
	$str =~ s/\s+$//;
	$str = lc($str);
	$pair_hash{$str} = $set;
    }
    close(I);
}

### 2. read mturk file
my %pair_set;
my %pair_score;
my %pair_count;
my %all_score;
my $count = 0;
open(I, $input_file) || die "Cannot open $input_file\n";
while (my $str = <I>) {
    $str =~ s/\s+$//;
    $str =~ s/^(\d+)\t//;
    $str = lc($str);
    my $score = $1;
    if (!exists $pair_hash{$str}) {
	#print "$score\t$str\n";
	$count++;
	next;
    }
    $all_score{$str} .= "$score ";
    $pair_score{$str} += $score;
    $pair_count{$str}++;
    $pair_set{$str} = $pair_hash{$str};
}
close(I);

#print "$count\n";


### 3. write into file
my $count2 = 0;
my $count3 = 0;
open(O, ">$output_file") || die "Cannot create $output_file\n";
foreach my $pair (sort {$pair_set{$a} cmp $pair_set{$b}} keys %pair_set) {
    my $score = $pair_score{$pair} / $pair_count{$pair};
    my $as = $all_score{$pair};
    $as =~ s/\s+$//;
    my @ss = split(/\s+/, $as);
    if (@ss > 5) {
	$count2++;
	splice(@ss, 5);
    }
    if (@ss < 5) {
	#print "$pair_count{$pair}\t$pair\n";
	$count3++;
    }
    my $as_str = join(" ", @ss);
    print O "$score\t$pair_count{$pair}\t$pair_set{$pair}\t$as_str\t$pair\n";
}
close(O);

#print "more than 5 annotations: $count2\n";
#print "less than 5 annotations: $count3\n";
