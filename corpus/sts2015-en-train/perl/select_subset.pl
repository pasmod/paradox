#-----------------------------------------------------------------------------
# select a subset with higher annotator agreement and a reasonable difficulty
#-----------------------------------------------------------------------------

use strict;
use warnings;
use Statistics::Basic qw(:all);
use String::Similarity;
#use Math::Random;

die "Usage:  perl  select_subset.pl  input_file  filter_dir\n" if (@ARGV != 2);

#my $input_file = "../data/clean/text.clean";
#my $output_dir = "../data/test3";
my $input_file = $ARGV[0];
my $output_dir = $ARGV[1];

open(I, $input_file) || die "Cannot open $input_file\n";

srand(1234567);
### 1. read data
my %data;
while (my $str = <I>) {
    $str =~ s/\s+$//;
    my @info = split(/\t+/, $str);
    if (!exists $data{$info[2]}) {
	$data{$info[2]} = ();
    }
    push(@{$data{$info[2]}}, \@info);
}
close(I);

### 2. for each subset, sample data
foreach my $set (sort {$a cmp $b} keys %data) {
    my $subset = $data{$set};

    ### 3. bin data
    my @cat_data;
    foreach my $unit (@$subset) {
	my $s = $unit->[0];
	my $cat = 0;
	if ($s >= 4.5) {
	    $cat = 5;
	} elsif ($s >= 3.5 && $s < 4.5) {
	    $cat = 4;
	} elsif ($s >= 2.5 && $s < 3.5) {
	    $cat = 3;
	} elsif ($s >= 1.5 && $s < 2.5) {
	    $cat = 2;
	} elsif ($s >= 0.5 && $s < 1.5) {
	    $cat = 1;
	} else {
	    $cat = 0;
	}
	push(@{$cat_data[$cat]}, $unit);
    }
    
    
    ### 4. compute bin size
    my $total = 750;
    $total = 375 if ($set eq "answers-forums" || $set eq "belief");
    my $bin = &fill_bin(\@cat_data, $total);

    ### 5. sample data
    open(O, ">$output_dir/$set") || die "Cannot create $output_dir/$set\n";
    my $data = &sample_data($bin, \@cat_data);
    for (my $i = 0; $i < @$data; $i++) {
	my $cat = $data->[$i];
	for (my $j = 0; $j < @$cat; $j++) {
	    print O join("\t", @{$cat->[$j]}) . "\n";
	}
    }
    close(O);
    #die;

    ### 6. statistics
    my $avg_sd = 0;
    my $avg_diff = 0;
    for (my $i = 0; $i < @$subset; $i++) {
	$avg_sd += $subset->[$i]->[-5];
	$avg_diff += $subset->[$i]->[-2];
	#die "$data->[$i]->[-5]\t$data->[$i]->[-2]\n";
    }
    $avg_sd /= @$subset;
    $avg_diff /= @$subset;
    
    my $avg_sd2 = 0;
    my $avg_diff2 = 0;
    my $count = 0;
    for (my $i = 0; $i < @$data; $i++) {
	my $cat = $data->[$i];
	for (my $j = 0; $j < @$cat; $j++) {
	    $avg_sd2 += $cat->[$j]->[-5];
	    $avg_diff2 += $cat->[$j]->[-2];
	    $count++;
	}
    }
    $avg_sd2 /= $count;
    $avg_diff2 /= $count;

    my $avg_sd3 = 0;
    my $avg_diff3 = 0;
    for (my $i = 0; $i < @cat_data; $i++) {
        my $cat = $cat_data[$i];
	my $weight = scalar(@{$data->[$i]}) / @$cat;
	for (my $j = 0; $j < @$cat; $j++) {
            $avg_sd3 += $cat->[$j]->[-5] * $weight;
            $avg_diff3 += $cat->[$j]->[-2] * $weight;
        }
    }
    $avg_sd3 /= $count;
    $avg_diff3 /= $count;
    
    $avg_sd = sprintf("%.3f", $avg_sd);
    $avg_sd2 = sprintf("%.3f", $avg_sd2);
    $avg_sd3 = sprintf("%.3f", $avg_sd3);
    $avg_diff = sprintf("%.3f", $avg_diff);
    $avg_diff2 = sprintf("%.3f", $avg_diff2);
    $avg_diff3 = sprintf("%.3f", $avg_diff3);
    print "$set:\t$avg_sd\t$avg_diff\t$avg_sd3\t$avg_diff3\t$avg_sd2\t$avg_diff2\n";
    #die;

    print "$set: " . scalar(@$subset) . "\t$total\n";
    for (my $i = 0; $i < 6; $i++) {
	print "$i:\t" . scalar(@{$cat_data[$i]}) . "\t" . scalar(@{$data->[$i]}) . "\n";
    }

}




sub fill_bin() {
    my ($cat_data, $total) = @_;
    my @bin;
    my %size;
    for (my $i = 0; $i < @$cat_data; $i++) {
	$size{$i} = scalar(@{$cat_data->[$i]});
    }
    my @index = sort {$size{$a} <=> $size{$b}} keys %size;
    for (my $i = 0; $i < @index; $i++) {
	my $aim = int($total/(@index-$i));
	my $j = $index[$i];
	$bin[$j] = $aim;
	if ($aim > scalar(@{$cat_data->[$j]})) {
	    $bin[$j] = scalar(@{$cat_data->[$j]});
	}
	$total -= $bin[$j];
    }
    return \@bin;
}


sub sample_data() {
    my ($bin, $cat_data) = @_;
    my @data;

    # 1. compute score
    for (my $i = 0; $i < @$bin; $i++) {
	my $cat_d = $cat_data->[$i];
	my @gold_sims;
	my @str_sims;
	my @sds;
	for (my $j = 0; $j < @$cat_d; $j++) {
	    my @all_s = split(/\s+/, $cat_d->[$j]->[3]);
	    my $sd = stddev(@all_s);
	    push(@{$cat_d->[$j]}, $sd);
	    my $str_sim = similarity($cat_d->[$j]->[4], $cat_d->[$j]->[5]);
	    push(@{$cat_d->[$j]}, $str_sim);
	    
	    push(@sds, $sd);
	    push(@str_sims, $str_sim);
	    push(@gold_sims, $cat_d->[$j]->[0]);
	}
	my $sd_mean = mean(@sds);
	my $sd_std = stddev(@sds);
	my $gs_mean = mean(@gold_sims);
	my $ss_mean = mean(@str_sims);
	my $gs_std = stddev(@gold_sims);
	my $ss_std = stddev(@str_sims);

	#print "$i:\t$sd_mean\t$sd_std\n";
	if ($sd_std == 0 || $gs_std == 0 || $ss_std == 0) {
	    #print "\nstd is 0!!!\n";
	    for (my $j = 0; $j < @$cat_d; $j++) {
		push(@{$cat_d->[$j]}, 0);
		push(@{$cat_d->[$j]}, 0);
		push(@{$cat_d->[$j]}, 0);
	    }
	    print scalar(@$cat_d) . "\n\n";
	} else {
	    for (my $j = 0; $j < @$cat_d; $j++) {
		my $sd = ($cat_d->[$j]->[-2] - $sd_mean) / $sd_std;
		#$cat_d->[$j]->[-2] = $sd;
		my $gs = ($cat_d->[$j]->[0] - $gs_mean) / $sd_std;
		my $ss = ($cat_d->[$j]->[-1] - $ss_mean) / $ss_std;
		#$cat_d->[$j]->[-1] = $ss;
		my $diff = abs($gs-$ss);
		my $score = 1 + exp(-$sd) + exp($diff/2);
		push(@{$cat_d->[$j]}, $sd);
		push(@{$cat_d->[$j]}, $diff);
		push(@{$cat_d->[$j]}, $score);
	    }
	}
    }
    
    
    # 2. sample
    for (my $i = 0; $i < @$bin; $i++) {
	$data[$i] = ();
	my $cat_d = $cat_data->[$i];
	if ($bin->[$i] == @$cat_d) {
	    $data[$i] = $cat_d;
	} else {
#=head
	    my @interval;
	    $interval[0] = $cat_d->[0]->[-1];
	    for (my $j = 1; $j < @$cat_d; $j++) {
		$interval[$j] = $cat_d->[$j]->[-1] + $interval[$j-1];
	    }
	    for (my $j = 0; $j < @$cat_d; $j++) {
		$interval[$j] /= $interval[-1];
	    }
#=cut
	    my $count = 0;
	    my %used;
	    while ($count != $bin->[$i]) {
		my $rn = rand();
		my $k = 0;
		while ($rn > $interval[$k]) {
		    $k++;
		}
		if (exists $used{$k}) {
		    next;
		} else {
		    $used{$k} = 1;
		}
		push(@{$data[$i]}, $cat_d->[$k]);
		$count++;
	    }
	}
    }
    return \@data;
}
