######################################################################
package XML::DOM::NodeList;
######################################################################

use vars qw ( $EMPTY );

# Empty NodeList
$EMPTY = new XML::DOM::NodeList;

sub new 
{
    bless [], $_[0];
}

sub item 
{
    $_[0]->[$_[1]];
}

sub getLength 
{
    int (@{$_[0]});
}

#------------------------------------------------------------
# Extra method implementations

sub dispose
{
    my $self = shift;
    for my $kid (@{$self})
    {
	$kid->dispose;
    }
}

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    for my $kid (@{$self})
    { 
	$kid->setOwnerDocument ($doc);
    }
}

1; # package return code
