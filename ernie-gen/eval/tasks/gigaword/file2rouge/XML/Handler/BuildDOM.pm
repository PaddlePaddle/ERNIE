package XML::Handler::BuildDOM;
use strict;
use XML::DOM;

#
# TODO:
# - add support for parameter entity references
# - expand API: insert Elements in the tree or stuff into DocType etc.

sub new
{
    my ($class, %args) = @_;
    bless \%args, $class;
}

#-------- PerlSAX Handler methods ------------------------------

sub start_document # was Init
{
    my $self = shift;

    # Define Document if it's not set & not obtainable from Element or DocType
    $self->{Document} ||= 
	(defined $self->{Element} ? $self->{Element}->getOwnerDocument : undef)
     || (defined $self->{DocType} ? $self->{DocType}->getOwnerDocument : undef)
     || new XML::DOM::Document();

    $self->{Element} ||= $self->{Document};

    unless (defined $self->{DocType})
    {
	$self->{DocType} = $self->{Document}->getDoctype
	    if defined $self->{Document};

	unless (defined $self->{Doctype})
	{
#?? should be $doc->createDocType for extensibility!
	    $self->{DocType} = new XML::DOM::DocumentType ($self->{Document});
	    $self->{Document}->setDoctype ($self->{DocType});
	}
    }
  
    # Prepare for document prolog
    $self->{InProlog} = 1;

    # We haven't passed the root element yet
    $self->{EndDoc} = 0;

    undef $self->{LastText};
}

sub end_document # was Final
{
    my $self = shift;
    unless ($self->{SawDocType})
    {
	my $doctype = $self->{Document}->removeDoctype;
	$doctype->dispose;
#?? do we always want to destroy the Doctype?
    }
    $self->{Document};
}

sub characters # was Char
{
    my $self = $_[0];
    my $str = $_[1]->{Data};

    if ($self->{InCDATA} && $self->{KeepCDATA})
    {
	undef $self->{LastText};
	# Merge text with previous node if possible
	$self->{Element}->addCDATA ($str);
    }
    else
    {
	# Merge text with previous node if possible
	# Used to be:	$expat->{DOM_Element}->addText ($str);
	if ($self->{LastText})
	{
	    $self->{LastText}->appendData ($str);
	}
	else
	{
	    $self->{LastText} = $self->{Document}->createTextNode ($str);
	    $self->{Element}->appendChild ($self->{LastText});
	}
    }
}

sub start_element # was Start
{
    my ($self, $hash) = @_;
    my $elem = $hash->{Name};
    my $attr = $hash->{Attributes};

    my $parent = $self->{Element};
    my $doc = $self->{Document};
    
    if ($parent == $doc)
    {
	# End of document prolog, i.e. start of first Element
	$self->{InProlog} = 0;
    }
    
    undef $self->{LastText};
    my $node = $doc->createElement ($elem);
    $self->{Element} = $node;
    $parent->appendChild ($node);
    
    my $i = 0;
    my $n = scalar keys %$attr;
    return unless $n;

    if (exists $hash->{AttributeOrder})
    {
	my $defaulted = $hash->{Defaulted};
	my @order = @{ $hash->{AttributeOrder} };
	
	# Specified attributes
	for (my $i = 0; $i < $defaulted; $i++)
	{
	    my $a = $order[$i];
	    my $att = $doc->createAttribute ($a, $attr->{$a}, 1);
	    $node->setAttributeNode ($att);
	}

	# Defaulted attributes
	for (my $i = $defaulted; $i < @order; $i++)
	{
	    my $a = $order[$i];
	    my $att = $doc->createAttribute ($elem, $attr->{$a}, 0);
	    $node->setAttributeNode ($att);
	}
    }
    else
    {
	# We're assuming that all attributes were specified (1)
	for my $a (keys %$attr)
	{
	    my $att = $doc->createAttribute ($a, $attr->{$a}, 1);
	    $node->setAttributeNode ($att);
	}
    }
}

sub end_element
{
    my $self = shift;
    $self->{Element} = $self->{Element}->getParentNode;
    undef $self->{LastText};

    # Check for end of root element
    $self->{EndDoc} = 1 if ($self->{Element} == $self->{Document});
}

sub entity_reference # was Default
{
    my $self = $_[0];
    my $name = $_[1]->{Name};
    
    $self->{Element}->appendChild (
			    $self->{Document}->createEntityReference ($name));
    undef $self->{LastText};
}

sub start_cdata
{
    my $self = shift;
    $self->{InCDATA} = 1;
}

sub end_cdata
{
    my $self = shift;
    $self->{InCDATA} = 0;
}

sub comment
{
    my $self = $_[0];

    local $XML::DOM::IgnoreReadOnly = 1;

    undef $self->{LastText};
    my $comment = $self->{Document}->createComment ($_[1]->{Data});
    $self->{Element}->appendChild ($comment);
}

sub doctype_decl
{
    my ($self, $hash) = @_;

    $self->{DocType}->setParams ($hash->{Name}, $hash->{SystemId}, 
				 $hash->{PublicId}, $hash->{Internal});
    $self->{SawDocType} = 1;
}

sub attlist_decl
{
    my ($self, $hash) = @_;

    local $XML::DOM::IgnoreReadOnly = 1;

    $self->{DocType}->addAttDef ($hash->{ElementName},
				 $hash->{AttributeName},
				 $hash->{Type},
				 $hash->{Default},
				 $hash->{Fixed});
}

sub xml_decl
{
    my ($self, $hash) = @_;

    local $XML::DOM::IgnoreReadOnly = 1;

    undef $self->{LastText};
    $self->{Document}->setXMLDecl (new XML::DOM::XMLDecl ($self->{Document}, 
							  $hash->{Version},
							  $hash->{Encoding},
							  $hash->{Standalone}));
}

sub entity_decl
{
    my ($self, $hash) = @_;
    
    local $XML::DOM::IgnoreReadOnly = 1;

    # Parameter Entities names are passed starting with '%'
    my $parameter = 0;

#?? parameter entities currently not supported by PerlSAX!

    undef $self->{LastText};
    $self->{DocType}->addEntity ($parameter, $hash->{Name}, $hash->{Value}, 
				 $hash->{SystemId}, $hash->{PublicId}, 
				 $hash->{Notation});
}

# Unparsed is called when it encounters e.g:
#
#   <!ENTITY logo SYSTEM "http://server/logo.gif" NDATA gif>
#
sub unparsed_decl
{
    my ($self, $hash) = @_;

    local $XML::DOM::IgnoreReadOnly = 1;

    # same as regular ENTITY, as far as DOM is concerned
    $self->entity_decl ($hash);
}

sub element_decl
{
    my ($self, $hash) = @_;

    local $XML::DOM::IgnoreReadOnly = 1;

    undef $self->{LastText};
    $self->{DocType}->addElementDecl ($hash->{Name}, $hash->{Model});
}

sub notation_decl
{
    my ($self, $hash) = @_;

    local $XML::DOM::IgnoreReadOnly = 1;

    undef $self->{LastText};
    $self->{DocType}->addNotation ($hash->{Name}, $hash->{Base}, 
				   $hash->{SystemId}, $hash->{PublicId});
}

sub processing_instruction
{
    my ($self, $hash) = @_;

    local $XML::DOM::IgnoreReadOnly = 1;

    undef $self->{LastText};
    $self->{Element}->appendChild (new XML::DOM::ProcessingInstruction 
			    ($self->{Document}, $hash->{Target}, $hash->{Data}));
}

return 1;

__END__

=head1 NAME

XML::Handler::BuildDOM - PerlSAX handler that creates XML::DOM document structures

=head1 SYNOPSIS

 use XML::Handler::BuildDOM;
 use XML::Parser::PerlSAX;

 my $handler = new XML::Handler::BuildDOM (KeepCDATA => 1);
 my $parser = new XML::Parser::PerlSAX (Handler => $handler);

 my $doc = $parser->parsefile ("file.xml");

=head1 DESCRIPTION

XML::Handler::BuildDOM creates L<XML::DOM> document structures 
(i.e. L<XML::DOM::Document>) from PerlSAX events.

This class used to be called L<XML::PerlSAX::DOM> prior to libxml-enno 1.0.1.

=head2 CONSTRUCTOR OPTIONS

The XML::Handler::BuildDOM constructor supports the following options:

=over 4

=item * KeepCDATA => 1 

If set to 0 (default), CDATASections will be converted to regular text.

=item * Document => $doc

If undefined, start_document will extract it from Element or DocType (if set),
otherwise it will create a new XML::DOM::Document.

=item * Element => $elem

If undefined, it is set to Document. This will be the insertion point (or parent)
for the nodes defined by the following callbacks.

=item * DocType => $doctype

If undefined, start_document will extract it from Document (if possible).
Otherwise it adds a new XML::DOM::DocumentType to the Document.

=back
