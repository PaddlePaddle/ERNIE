package XML::DOM::PerlSAX;
use strict;

BEGIN
{
    if ($^W)
    {
	warn "XML::DOM::PerlSAX has been renamed to XML::Handler::BuildDOM, please modify your code accordingly.";
    }
}

use XML::Handler::BuildDOM;
use vars qw{ @ISA };
@ISA = qw{ XML::Handler::BuildDOM };

1; # package return code

__END__

=head1 NAME

XML::DOM::PerlSAX - Old name of L<XML::Handler::BuildDOM>

=head1 SYNOPSIS

 See L<XML::DOM::BuildDOM>

=head1 DESCRIPTION

XML::DOM::PerlSAX was renamed to L<XML::Handler::BuildDOM> to comply
with naming conventions for PerlSAX filters/handlers.

For backward compatibility, this package will remain in existence 
(it simply includes XML::Handler::BuildDOM), but it will print a warning when 
running with I<'perl -w'>.

=head1 AUTHOR

Enno Derksen is the original author.

Send bug reports, hints, tips, suggestions to T.J Mather at
<F<tjmather@tjmather.com>>.

=head1 SEE ALSO

L<XML::Handler::BuildDOM>, L<XML::DOM>

