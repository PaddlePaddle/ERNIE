######################################################################
package XML::DOM::NamedNodeMap;
######################################################################

use strict;

use Carp;
use XML::DOM::DOMException;
use XML::DOM::NodeList;

use vars qw( $Special );

# Constant definition:
# Note: a real Name should have at least 1 char, so nobody else should use this
$Special = "";

sub new 
{
    my ($class, %args) = @_;

    $args{Values} = new XML::DOM::NodeList;

    # Store all NamedNodeMap properties in element $Special
    bless { $Special => \%args}, $class;
}

sub getNamedItem 
{
    # Don't return the $Special item!
    ($_[1] eq $Special) ? undef : $_[0]->{$_[1]};
}

sub setNamedItem 
{
    my ($self, $node) = @_;
    my $prop = $self->{$Special};

    my $name = $node->getNodeName;

    if ($XML::DOM::SafeMode)
    {
	croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR)
	    if $self->isReadOnly;

	croak new XML::DOM::DOMException (WRONG_DOCUMENT_ERR)
	    if $node->[XML::DOM::Node::_Doc] != $prop->{Doc};

	croak new XML::DOM::DOMException (INUSE_ATTRIBUTE_ERR)
	    if defined ($node->[XML::DOM::Node::_UsedIn]);

	croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR,
		      "can't add name with NodeName [$name] to NamedNodeMap")
	    if $name eq $Special;
    }

    my $values = $prop->{Values};
    my $index = -1;

    my $prev = $self->{$name};
    if (defined $prev)
    {
	# decouple previous node
	$prev->decoupleUsedIn;

	# find index of $prev
	$index = 0;
	for my $val (@{$values})
	{
	    last if ($val == $prev);
	    $index++;
	}
    }

    $self->{$name} = $node;    
    $node->[XML::DOM::Node::_UsedIn] = $self;

    if ($index == -1)
    {
	push (@{$values}, $node);
    }
    else	# replace previous node with new node
    {
	splice (@{$values}, $index, 1, $node);
    }
    
    $prev;
}

sub removeNamedItem 
{
    my ($self, $name) = @_;

    # Be careful that user doesn't delete $Special node!
    croak new XML::DOM::DOMException (NOT_FOUND_ERR)
        if $name eq $Special;

    my $node = $self->{$name};

    croak new XML::DOM::DOMException (NOT_FOUND_ERR)
        unless defined $node;

    # The DOM Spec doesn't mention this Exception - I think it's an oversight
    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR)
	if $self->isReadOnly;

    $node->decoupleUsedIn;
    delete $self->{$name};

    # remove node from Values list
    my $values = $self->getValues;
    my $index = 0;
    for my $val (@{$values})
    {
	if ($val == $node)
	{
	    splice (@{$values}, $index, 1, ());
	    last;
	}
	$index++;
    }
    $node;
}

# The following 2 are really bogus. DOM should use an iterator instead (Clark)

sub item 
{
    my ($self, $item) = @_;
    $self->{$Special}->{Values}->[$item];
}

sub getLength 
{
    my ($self) = @_;
    my $vals = $self->{$Special}->{Values};
    int (@$vals);
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    return 0 if $XML::DOM::IgnoreReadOnly;

    my $used = $_[0]->{$Special}->{UsedIn};
    defined $used ? $used->isReadOnly : 0;
}

sub cloneNode
{
    my ($self, $deep) = @_;
    my $prop = $self->{$Special};

    my $map = new XML::DOM::NamedNodeMap (Doc => $prop->{Doc});
    # Not copying Parent property on purpose! 

    local $XML::DOM::IgnoreReadOnly = 1;	# temporarily...

    for my $val (@{$prop->{Values}})
    {
	my $key = $val->getNodeName;

	my $newNode = $val->cloneNode ($deep);
	$newNode->[XML::DOM::Node::_UsedIn] = $map;
	$map->{$key} = $newNode;
	push (@{$map->{$Special}->{Values}}, $newNode);
    }

    $map;
}

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    my $special = $self->{$Special};

    $special->{Doc} = $doc;
    for my $kid (@{$special->{Values}})
    {
	$kid->setOwnerDocument ($doc);
    }
}

sub getChildIndex
{
    my ($self, $attr) = @_;
    my $i = 0;
    for my $kid (@{$self->{$Special}->{Values}})
    {
	return $i if $kid == $attr;
	$i++;
    }
    -1;	# not found
}

sub getValues
{
    wantarray ? @{ $_[0]->{$Special}->{Values} } : $_[0]->{$Special}->{Values};
}

# Remove circular dependencies. The NamedNodeMap and its values should
# not be used afterwards.
sub dispose
{
    my $self = shift;

    for my $kid (@{$self->getValues})
    {
	undef $kid->[XML::DOM::Node::_UsedIn]; # was delete
	$kid->dispose;
    }

    delete $self->{$Special}->{Doc};
    delete $self->{$Special}->{Parent};
    delete $self->{$Special}->{Values};

    for my $key (keys %$self)
    {
	delete $self->{$key};
    }
}

sub setParentNode
{
    $_[0]->{$Special}->{Parent} = $_[1];
}

sub getProperty
{
    $_[0]->{$Special}->{$_[1]};
}

#?? remove after debugging
sub toString
{
    my ($self) = @_;
    my $str = "NamedNodeMap[";
    while (my ($key, $val) = each %$self)
    {
	if ($key eq $Special)
	{
	    $str .= "##Special (";
	    while (my ($k, $v) = each %$val)
	    {
		if ($k eq "Values")
		{
		    $str .= $k . " => [";
		    for my $a (@$v)
		    {
#			$str .= $a->getNodeName . "=" . $a . ",";
			$str .= $a->toString . ",";
		    }
		    $str .= "], ";
		}
		else
		{
		    $str .= $k . " => " . $v . ", ";
		}
	    }
	    $str .= "), ";
	}
	else
	{
	    $str .= $key . " => " . $val . ", ";
	}
    }
    $str . "]";
}

1; # package return code
