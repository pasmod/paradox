use strict;
use warnings;
use Text::CSV;

die "Usage:  perl  read_csv.pl  mturk_dir  output_file\n" if (@ARGV != 2);

#my $input_dir = "../data/raw/mturk";
#my $output_file = "../data/clean/text";
my $input_dir = $ARGV[0];
my $output_file = $ARGV[1];

open(O, ">$output_file") || die "Cannot create $output_file\n";

my $csv = Text::CSV->new({sep_char => ','});

my $count = 0;
my $j = 0;
my $no_r = 0;
foreach my $input_file (<$input_dir/*.csv>) {
    print "$input_file\n";
    open(I, $input_file) || die "Cannot open $input_file\n";
    <I>; # ignore 1st line
    
    my $offset = 0;
    $offset = 1 if ($input_file =~ /1775938/ || $input_file =~ /1775939/);

    while (my $str = <I>) {
	$str =~ s/\s+$//;
	$j++;
	next if ($str eq "");
	#die if ($j > 10);



	### 1. if it can be handled by Text::CSV
	my @fields;
	if ($csv->parse($str)) {
	    @fields = $csv->fields();
	    
	} else {
	### 2. otherwise, it will be handled manually!!!omg!!!
	    #warn "$str\n";
	    @fields = split("\",\"", $str);
	    $count++;
	    #print "$fields[27]\n$fields[28]\n";
	    #die "$str\n";
	}

	for (my $i = 0; $i < 5; $i++) {
	    my $rating_str = $fields[$i+37+$offset];
	    if (!defined $rating_str) {
		die "$str\n";
	    }
	    #print "$rating\n";
	    $rating_str =~ /^\((\d)\)/;
	    my $rating = $1;
	    if (!defined $rating) {
		#print "$str\n";
		#warn "$rating_str\n";

		$no_r++;
	    } else {
		print O $rating . "\t" . $fields[$i*2+27] . "\t" . $fields[$i*2+28] . "\n";
	    }
	}
    }
    #die;
    close(I);
}
#print "$count\n";
#print "no rating: $no_r\n";
