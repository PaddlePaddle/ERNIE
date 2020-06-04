################################################################################
#
# Perl module: XML::DOM
#
# By Enno Derksen
#
################################################################################
#
# To do:
#
# * optimize Attr if it only contains 1 Text node to hold the value
# * fix setDocType!
#
# * BUG: setOwnerDocument - does not process default attr values correctly,
#   they still point to the old doc.
# * change Exception mechanism
# * maybe: more checking of sysId etc.
# * NoExpand mode (don't know what else is useful)
# * various odds and ends: see comments starting with "??"
# * normalize(1) could also expand CDataSections and EntityReferences
# * parse a DocumentFragment?
# * encoding support
#
######################################################################

######################################################################
package XML::DOM;
######################################################################

use strict;

use vars qw( $VERSION @ISA @EXPORT
	     $IgnoreReadOnly $SafeMode $TagStyle
	     %DefaultEntities %DecodeDefaultEntity
	   );
use Carp;
use XML::RegExp;

BEGIN
{
    require XML::Parser;
    $VERSION = '1.44';

    my $needVersion = '2.28';
    die "need at least XML::Parser version $needVersion (current=${XML::Parser::VERSION})"
	unless $XML::Parser::VERSION >= $needVersion;

    @ISA = qw( Exporter );

    # Constants for XML::DOM Node types
    @EXPORT = qw(
	     UNKNOWN_NODE
	     ELEMENT_NODE
	     ATTRIBUTE_NODE
	     TEXT_NODE
	     CDATA_SECTION_NODE
	     ENTITY_REFERENCE_NODE
	     ENTITY_NODE
	     PROCESSING_INSTRUCTION_NODE
	     COMMENT_NODE
	     DOCUMENT_NODE
	     DOCUMENT_TYPE_NODE
	     DOCUMENT_FRAGMENT_NODE
	     NOTATION_NODE
	     ELEMENT_DECL_NODE
	     ATT_DEF_NODE
	     XML_DECL_NODE
	     ATTLIST_DECL_NODE
	    );
}

#---- Constant definitions

# Node types

sub UNKNOWN_NODE                () { 0 }		# not in the DOM Spec

sub ELEMENT_NODE                () { 1 }
sub ATTRIBUTE_NODE              () { 2 }
sub TEXT_NODE                   () { 3 }
sub CDATA_SECTION_NODE          () { 4 }
sub ENTITY_REFERENCE_NODE       () { 5 }
sub ENTITY_NODE                 () { 6 }
sub PROCESSING_INSTRUCTION_NODE () { 7 }
sub COMMENT_NODE                () { 8 }
sub DOCUMENT_NODE               () { 9 }
sub DOCUMENT_TYPE_NODE          () { 10}
sub DOCUMENT_FRAGMENT_NODE      () { 11}
sub NOTATION_NODE               () { 12}

sub ELEMENT_DECL_NODE		() { 13 }	# not in the DOM Spec
sub ATT_DEF_NODE 		() { 14 }	# not in the DOM Spec
sub XML_DECL_NODE 		() { 15 }	# not in the DOM Spec
sub ATTLIST_DECL_NODE		() { 16 }	# not in the DOM Spec

%DefaultEntities = 
(
 "quot"		=> '"',
 "gt"		=> ">",
 "lt"		=> "<",
 "apos"		=> "'",
 "amp"		=> "&"
);

%DecodeDefaultEntity =
(
 '"' => "&quot;",
 ">" => "&gt;",
 "<" => "&lt;",
 "'" => "&apos;",
 "&" => "&amp;"
);

#
# If you don't want DOM warnings to use 'warn', override this method like this:
#
# { # start block scope
#	local *XML::DOM::warning = \&my_warn;
#	... your code here ...
# } # end block scope (old XML::DOM::warning takes effect again)
#
sub warning	# static
{
    warn @_;
}

#
# This method defines several things in the caller's package, so you can use named constants to
# access the array that holds the member data, i.e. $self->[_Data]. It assumes the caller's package
# defines a class that is implemented as a blessed array reference.
# Note that this is very similar to using 'use fields' and 'use base'.
#
# E.g. if $fields eq "Name Model", $parent eq "XML::DOM::Node" and
# XML::DOM::Node had "A B C" as fields and it was called from package "XML::DOM::ElementDecl",
# then this code would basically do the following:
#
# package XML::DOM::ElementDecl;
#
# sub _Name  () { 3 }	# Note that parent class had three fields
# sub _Model () { 4 }
#
# # Maps constant names (without '_') to constant (int) value
# %HFIELDS = ( %XML::DOM::Node::HFIELDS, Name => _Name, Model => _Model );
#
# # Define XML:DOM::ElementDecl as a subclass of XML::DOM::Node
# @ISA = qw{ XML::DOM::Node };
#
# # The following function names can be exported into the user's namespace.
# @EXPORT_OK = qw{ _Name _Model };
#
# # The following function names can be exported into the user's namespace
# # with: import XML::DOM::ElementDecl qw( :Fields );
# %EXPORT_TAGS = ( Fields => qw{ _Name _Model } );
#
sub def_fields	# static
{
    my ($fields, $parent) = @_;

    my ($pkg) = caller;

    no strict 'refs';

    my @f = split (/\s+/, $fields);
    my $n = 0;

    my %hfields;
    if (defined $parent)
    {
	my %pf = %{"$parent\::HFIELDS"};
	%hfields = %pf;

	$n = scalar (keys %pf);
	@{"$pkg\::ISA"} = ( $parent );
    }

    my $i = $n;
    for (@f)
    {
	eval "sub $pkg\::_$_ () { $i }";
	$hfields{$_} = $i;
	$i++;
    }
    %{"$pkg\::HFIELDS"} = %hfields;
    @{"$pkg\::EXPORT_OK"} = map { "_$_" } @f;
    
    ${"$pkg\::EXPORT_TAGS"}{Fields} = [ map { "_$_" } @f ];
}

# sub blesh
# {
#     my $hashref = shift;
#     my $class = shift;
#     no strict 'refs';
#     my $self = bless [\%{"$class\::FIELDS"}], $class;
#     if (defined $hashref)
#     {
# 	for (keys %$hashref)
# 	{
# 	    $self->{$_} = $hashref->{$_};
# 	}
#     }
#     $self;
# }

# sub blesh2
# {
#     my $hashref = shift;
#     my $class = shift;
#     no strict 'refs';
#     my $self = bless [\%{"$class\::FIELDS"}], $class;
#     if (defined $hashref)
#     {
# 	for (keys %$hashref)
# 	{
# 	    eval { $self->{$_} = $hashref->{$_}; };
# 	    croak "ERROR in field [$_] $@" if $@;
# 	}
#     }
#     $self;
#}

#
# CDATA section may not contain "]]>"
#
sub encodeCDATA
{
    my ($str) = shift;
    $str =~ s/]]>/]]&gt;/go;
    $str;
}

#
# PI may not contain "?>"
#
sub encodeProcessingInstruction
{
    my ($str) = shift;
    $str =~ s/\?>/?&gt;/go;
    $str;
}

#
#?? Not sure if this is right - must prevent double minus somehow...
#
sub encodeComment
{
    my ($str) = shift;
    return undef unless defined $str;

    $str =~ s/--/&#45;&#45;/go;
    $str;
}

#
# For debugging
#
sub toHex
{
    my $str = shift;
    my $len = length($str);
    my @a = unpack ("C$len", $str);
    my $s = "";
    for (@a)
    {
	$s .= sprintf ("%02x", $_);
    }
    $s;
}

#
# 2nd parameter $default: list of Default Entity characters that need to be 
# converted (e.g. "&<" for conversion to "&amp;" and "&lt;" resp.)
#
sub encodeText
{
    my ($str, $default) = @_;
    return undef unless defined $str;

    if ($] >= 5.006) {
      $str =~ s/([$default])|(]]>)/
        defined ($1) ? $DecodeDefaultEntity{$1} : "]]&gt;" /egs;
    }
    else {
      $str =~ s/([\xC0-\xDF].|[\xE0-\xEF]..|[\xF0-\xFF]...)|([$default])|(]]>)/
        defined($1) ? XmlUtf8Decode ($1) :
        defined ($2) ? $DecodeDefaultEntity{$2} : "]]&gt;" /egs;
    }

#?? could there be references that should not be expanded?
# e.g. should not replace &#nn; &#xAF; and &abc;
#    $str =~ s/&(?!($ReName|#[0-9]+|#x[0-9a-fA-F]+);)/&amp;/go;

    $str;
}

#
# Used by AttDef - default value
#
sub encodeAttrValue
{
    encodeText (shift, '"&<>');
}

#
# Converts an integer (Unicode - ISO/IEC 10646) to a UTF-8 encoded character 
# sequence.
# Used when converting e.g. &#123; or &#x3ff; to a string value.
#
# Algorithm borrowed from expat/xmltok.c/XmlUtf8Encode()
#
# not checking for bad characters: < 0, x00-x08, x0B-x0C, x0E-x1F, xFFFE-xFFFF
#
sub XmlUtf8Encode
{
    my $n = shift;
    if ($n < 0x80)
    {
	return chr ($n);
    }
    elsif ($n < 0x800)
    {
	return pack ("CC", (($n >> 6) | 0xc0), (($n & 0x3f) | 0x80));
    }
    elsif ($n < 0x10000)
    {
	return pack ("CCC", (($n >> 12) | 0xe0), ((($n >> 6) & 0x3f) | 0x80),
		     (($n & 0x3f) | 0x80));
    }
    elsif ($n < 0x110000)
    {
	return pack ("CCCC", (($n >> 18) | 0xf0), ((($n >> 12) & 0x3f) | 0x80),
		     ((($n >> 6) & 0x3f) | 0x80), (($n & 0x3f) | 0x80));
    }
    croak "number is too large for Unicode [$n] in &XmlUtf8Encode";
}

#
# Opposite of XmlUtf8Decode plus it adds prefix "&#" or "&#x" and suffix ";"
# The 2nd parameter ($hex) indicates whether the result is hex encoded or not.
#
sub XmlUtf8Decode
{
    my ($str, $hex) = @_;
    my $len = length ($str);
    my $n;

    if ($len == 2)
    {
	my @n = unpack "C2", $str;
	$n = (($n[0] & 0x3f) << 6) + ($n[1] & 0x3f);
    }
    elsif ($len == 3)
    {
	my @n = unpack "C3", $str;
	$n = (($n[0] & 0x1f) << 12) + (($n[1] & 0x3f) << 6) + 
		($n[2] & 0x3f);
    }
    elsif ($len == 4)
    {
	my @n = unpack "C4", $str;
	$n = (($n[0] & 0x0f) << 18) + (($n[1] & 0x3f) << 12) + 
		(($n[2] & 0x3f) << 6) + ($n[3] & 0x3f);
    }
    elsif ($len == 1)	# just to be complete...
    {
	$n = ord ($str);
    }
    else
    {
	croak "bad value [$str] for XmlUtf8Decode";
    }
    $hex ? sprintf ("&#x%x;", $n) : "&#$n;";
}

$IgnoreReadOnly = 0;
$SafeMode = 1;

sub getIgnoreReadOnly
{
    $IgnoreReadOnly;
}

#
# The global flag $IgnoreReadOnly is set to the specified value and the old 
# value of $IgnoreReadOnly is returned.
#
# To temporarily disable read-only related exceptions (i.e. when parsing
# XML or temporarily), do the following:
#
# my $oldIgnore = XML::DOM::ignoreReadOnly (1);
# ... do whatever you want ...
# XML::DOM::ignoreReadOnly ($oldIgnore);
#
sub ignoreReadOnly
{
    my $i = $IgnoreReadOnly;
    $IgnoreReadOnly = $_[0];
    return $i;
}

#
# XML spec seems to break its own rules... (see ENTITY xmlpio)
#
sub forgiving_isValidName
{
    use bytes;  # XML::RegExp expressed in terms encoded UTF8
    $_[0] =~ /^$XML::RegExp::Name$/o;
}

#
# Don't allow names starting with xml (either case)
#
sub picky_isValidName
{
    use bytes;  # XML::RegExp expressed in terms encoded UTF8
    $_[0] =~ /^$XML::RegExp::Name$/o and $_[0] !~ /^xml/i;
}

# Be forgiving by default, 
*isValidName = \&forgiving_isValidName;

sub allowReservedNames		# static
{
    *isValidName = ($_[0] ? \&forgiving_isValidName : \&picky_isValidName);
}

sub getAllowReservedNames	# static
{
    *isValidName == \&forgiving_isValidName;
}

#
# Always compress empty tags by default
# This is used by Element::print.
#
$TagStyle = sub { 0 };

sub setTagCompression
{
    $TagStyle = shift;
}

######################################################################
package XML::DOM::PrintToFileHandle;
######################################################################

#
# Used by XML::DOM::Node::printToFileHandle
#

sub new
{
    my($class, $fn) = @_;
    bless $fn, $class;
}

sub print
{
    my ($self, $str) = @_;
    print $self $str;
}

######################################################################
package XML::DOM::PrintToString;
######################################################################

use vars qw{ $Singleton };

#
# Used by XML::DOM::Node::toString to concatenate strings
#

sub new
{
    my($class) = @_;
    my $str = "";
    bless \$str, $class;
}

sub print
{
    my ($self, $str) = @_;
    $$self .= $str;
}

sub toString
{
    my $self = shift;
    $$self;
}

sub reset
{
    ${$_[0]} = "";
}

$Singleton = new XML::DOM::PrintToString;

######################################################################
package XML::DOM::DOMImplementation;
######################################################################
 
$XML::DOM::DOMImplementation::Singleton =
  bless \$XML::DOM::DOMImplementation::Singleton, 'XML::DOM::DOMImplementation';
 
sub hasFeature 
{
    my ($self, $feature, $version) = @_;
 
    uc($feature) eq 'XML' and ($version eq '1.0' || $version eq '');
}


######################################################################
package XML::XQL::Node;		# forward declaration
######################################################################

######################################################################
package XML::DOM::Node;
######################################################################

use vars qw( @NodeNames @EXPORT @ISA %HFIELDS @EXPORT_OK @EXPORT_TAGS );

BEGIN 
{
  use XML::DOM::DOMException;
  import Carp;

  require FileHandle;

  @ISA = qw( Exporter XML::XQL::Node );

  # NOTE: SortKey is used in XML::XQL::Node. 
  #       UserData is reserved for users (Hang your data here!)
  XML::DOM::def_fields ("C A Doc Parent ReadOnly UsedIn Hidden SortKey UserData");

  push (@EXPORT, qw(
		    UNKNOWN_NODE
		    ELEMENT_NODE
		    ATTRIBUTE_NODE
		    TEXT_NODE
		    CDATA_SECTION_NODE
		    ENTITY_REFERENCE_NODE
		    ENTITY_NODE
		    PROCESSING_INSTRUCTION_NODE
		    COMMENT_NODE
		    DOCUMENT_NODE
		    DOCUMENT_TYPE_NODE
		    DOCUMENT_FRAGMENT_NODE
		    NOTATION_NODE
		    ELEMENT_DECL_NODE
		    ATT_DEF_NODE
		    XML_DECL_NODE
		    ATTLIST_DECL_NODE
		   ));
}

#---- Constant definitions

# Node types

sub UNKNOWN_NODE                () {0;}		# not in the DOM Spec

sub ELEMENT_NODE                () {1;}
sub ATTRIBUTE_NODE              () {2;}
sub TEXT_NODE                   () {3;}
sub CDATA_SECTION_NODE          () {4;}
sub ENTITY_REFERENCE_NODE       () {5;}
sub ENTITY_NODE                 () {6;}
sub PROCESSING_INSTRUCTION_NODE () {7;}
sub COMMENT_NODE                () {8;}
sub DOCUMENT_NODE               () {9;}
sub DOCUMENT_TYPE_NODE          () {10;}
sub DOCUMENT_FRAGMENT_NODE      () {11;}
sub NOTATION_NODE               () {12;}

sub ELEMENT_DECL_NODE		() {13;}	# not in the DOM Spec
sub ATT_DEF_NODE 		() {14;}	# not in the DOM Spec
sub XML_DECL_NODE 		() {15;}	# not in the DOM Spec
sub ATTLIST_DECL_NODE		() {16;}	# not in the DOM Spec

@NodeNames = (
	      "UNKNOWN_NODE",	# not in the DOM Spec!

	      "ELEMENT_NODE",
	      "ATTRIBUTE_NODE",
	      "TEXT_NODE",
	      "CDATA_SECTION_NODE",
	      "ENTITY_REFERENCE_NODE",
	      "ENTITY_NODE",
	      "PROCESSING_INSTRUCTION_NODE",
	      "COMMENT_NODE",
	      "DOCUMENT_NODE",
	      "DOCUMENT_TYPE_NODE",
	      "DOCUMENT_FRAGMENT_NODE",
	      "NOTATION_NODE",

	      "ELEMENT_DECL_NODE",
	      "ATT_DEF_NODE",
	      "XML_DECL_NODE",
	      "ATTLIST_DECL_NODE"
	     );

sub decoupleUsedIn
{
    my $self = shift;
    undef $self->[_UsedIn]; # was delete
}

sub getParentNode
{
    $_[0]->[_Parent];
}

sub appendChild
{
    my ($self, $node) = @_;

    # REC 7473
    if ($XML::DOM::SafeMode)
    {
	croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
					  "node is ReadOnly")
	    if $self->isReadOnly;
    }

    my $doc = $self->[_Doc];

    if ($node->isDocumentFragmentNode)
    {
	if ($XML::DOM::SafeMode)
	{
	    for my $n (@{$node->[_C]})
	    {
		croak new XML::DOM::DOMException (WRONG_DOCUMENT_ERR,
						  "nodes belong to different documents")
		    if $doc != $n->[_Doc];
		
		croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
						  "node is ancestor of parent node")
		    if $n->isAncestor ($self);
		
		croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
						  "bad node type")
		    if $self->rejectChild ($n);
	    }
	}

	my @list = @{$node->[_C]};	# don't try to compress this
	for my $n (@list)
	{
	    $n->setParentNode ($self);
	}
	push @{$self->[_C]}, @list;
    }
    else
    {
	if ($XML::DOM::SafeMode)
	{
	    croak new XML::DOM::DOMException (WRONG_DOCUMENT_ERR,
						  "nodes belong to different documents")
		if $doc != $node->[_Doc];
		
	    croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
						  "node is ancestor of parent node")
		if $node->isAncestor ($self);
		
	    croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
						  "bad node type")
		if $self->rejectChild ($node);
	}
	$node->setParentNode ($self);
	push @{$self->[_C]}, $node;
    }
    $node;
}

sub getChildNodes
{
    # NOTE: if node can't have children, $self->[_C] is undef.
    my $kids = $_[0]->[_C];

    # Return a list if called in list context.
    wantarray ? (defined ($kids) ? @{ $kids } : ()) :
	        (defined ($kids) ? $kids : $XML::DOM::NodeList::EMPTY);
}

sub hasChildNodes
{
    my $kids = $_[0]->[_C];
    defined ($kids) && @$kids > 0;
}

# This method is overriden in Document
sub getOwnerDocument
{
    $_[0]->[_Doc];
}

sub getFirstChild
{
    my $kids = $_[0]->[_C];
    defined $kids ? $kids->[0] : undef; 
}

sub getLastChild
{
    my $kids = $_[0]->[_C];
    defined $kids ? $kids->[-1] : undef; 
}

sub getPreviousSibling
{
    my $self = shift;

    my $pa = $self->[_Parent];
    return undef unless $pa;
    my $index = $pa->getChildIndex ($self);
    return undef unless $index;

    $pa->getChildAtIndex ($index - 1);
}

sub getNextSibling
{
    my $self = shift;

    my $pa = $self->[_Parent];
    return undef unless $pa;

    $pa->getChildAtIndex ($pa->getChildIndex ($self) + 1);
}

sub insertBefore
{
    my ($self, $node, $refNode) = @_;

    return $self->appendChild ($node) unless $refNode;	# append at the end

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    my @nodes = ($node);
    @nodes = @{$node->[_C]}
	if $node->getNodeType == DOCUMENT_FRAGMENT_NODE;

    my $doc = $self->[_Doc];

    for my $n (@nodes)
    {
	croak new XML::DOM::DOMException (WRONG_DOCUMENT_ERR,
					  "nodes belong to different documents")
	    if $doc != $n->[_Doc];
	
	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "node is ancestor of parent node")
	    if $n->isAncestor ($self);

	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "bad node type")
	    if $self->rejectChild ($n);
    }
    my $index = $self->getChildIndex ($refNode);

    croak new XML::DOM::DOMException (NOT_FOUND_ERR,
				      "reference node not found")
	if $index == -1;

    for my $n (@nodes)
    {
	$n->setParentNode ($self);
    }

    splice (@{$self->[_C]}, $index, 0, @nodes);
    $node;
}

sub replaceChild
{
    my ($self, $node, $refNode) = @_;

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    my @nodes = ($node);
    @nodes = @{$node->[_C]}
	if $node->getNodeType == DOCUMENT_FRAGMENT_NODE;

    for my $n (@nodes)
    {
	croak new XML::DOM::DOMException (WRONG_DOCUMENT_ERR,
					  "nodes belong to different documents")
	    if $self->[_Doc] != $n->[_Doc];

	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "node is ancestor of parent node")
	    if $n->isAncestor ($self);

	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "bad node type")
	    if $self->rejectChild ($n);
    }

    my $index = $self->getChildIndex ($refNode);
    croak new XML::DOM::DOMException (NOT_FOUND_ERR,
				      "reference node not found")
	if $index == -1;

    for my $n (@nodes)
    {
	$n->setParentNode ($self);
    }
    splice (@{$self->[_C]}, $index, 1, @nodes);

    $refNode->removeChildHoodMemories;
    $refNode;
}

sub removeChild
{
    my ($self, $node) = @_;

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    my $index = $self->getChildIndex ($node);

    croak new XML::DOM::DOMException (NOT_FOUND_ERR,
				      "reference node not found")
	if $index == -1;

    splice (@{$self->[_C]}, $index, 1, ());

    $node->removeChildHoodMemories;
    $node;
}

# Merge all subsequent Text nodes in this subtree
sub normalize
{
    my ($self) = shift;
    my $prev = undef;	# previous Text node

    return unless defined $self->[_C];

    my @nodes = @{$self->[_C]};
    my $i = 0;
    my $n = @nodes;
    while ($i < $n)
    {
	my $node = $self->getChildAtIndex($i);
	my $type = $node->getNodeType;

	if (defined $prev)
	{
	    # It should not merge CDATASections. Dom Spec says:
	    #  Adjacent CDATASections nodes are not merged by use
	    #  of the Element.normalize() method.
	    if ($type == TEXT_NODE)
	    {
		$prev->appendData ($node->getData);
		$self->removeChild ($node);
		$i--;
		$n--;
	    }
	    else
	    {
		$prev = undef;
		if ($type == ELEMENT_NODE)
		{
		    $node->normalize;
		    if (defined $node->[_A])
		    {
			for my $attr (@{$node->[_A]->getValues})
			{
			    $attr->normalize;
			}
		    }
		}
	    }
	}
	else
	{
	    if ($type == TEXT_NODE)
	    {
		$prev = $node;
	    }
	    elsif ($type == ELEMENT_NODE)
	    {
		$node->normalize;
		if (defined $node->[_A])
		{
		    for my $attr (@{$node->[_A]->getValues})
		    {
			$attr->normalize;
		    }
		}
	    }
	}
	$i++;
    }
}

#
# Return all Element nodes in the subtree that have the specified tagName.
# If tagName is "*", all Element nodes are returned.
# NOTE: the DOM Spec does not specify a 3rd or 4th parameter
#
sub getElementsByTagName
{
    my ($self, $tagName, $recurse, $list) = @_;
    $recurse = 1 unless defined $recurse;
    $list = (wantarray ? [] : new XML::DOM::NodeList) unless defined $list;

    return unless defined $self->[_C];

    # preorder traversal: check parent node first
    for my $kid (@{$self->[_C]})
    {
	if ($kid->isElementNode)
	{
	    if ($tagName eq "*" || $tagName eq $kid->getTagName)
	    {
		push @{$list}, $kid;
	    }
	    $kid->getElementsByTagName ($tagName, $recurse, $list) if $recurse;
	}
    }
    wantarray ? @{ $list } : $list;
}

sub getNodeValue
{
    undef;
}

sub setNodeValue
{
    # no-op
}

#
# Redefined by XML::DOM::Element
#
sub getAttributes
{
    undef;
}

#------------------------------------------------------------
# Extra method implementations

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    $self->[_Doc] = $doc;

    return unless defined $self->[_C];

    for my $kid (@{$self->[_C]})
    {
	$kid->setOwnerDocument ($doc);
    }
}

sub cloneChildren
{
    my ($self, $node, $deep) = @_;
    return unless $deep;
    
    return unless defined $self->[_C];

    local $XML::DOM::IgnoreReadOnly = 1;

    for my $kid (@{$node->[_C]})
    {
	my $newNode = $kid->cloneNode ($deep);
	push @{$self->[_C]}, $newNode;
	$newNode->setParentNode ($self);
    }
}

#
# For internal use only!
#
sub removeChildHoodMemories
{
    my ($self) = @_;

    undef $self->[_Parent]; # was delete
}

#
# Remove circular dependencies. The Node and its children should
# not be used afterwards.
#
sub dispose
{
    my $self = shift;

    $self->removeChildHoodMemories;

    if (defined $self->[_C])
    {
	$self->[_C]->dispose;
	undef $self->[_C]; # was delete
    }
    undef $self->[_Doc]; # was delete
}

#
# For internal use only!
#
sub setParentNode
{
    my ($self, $parent) = @_;

    # REC 7473
    my $oldParent = $self->[_Parent];
    if (defined $oldParent)
    {
	# remove from current parent
	my $index = $oldParent->getChildIndex ($self);

	# NOTE: we don't have to check if [_C] is defined,
	# because were removing a child here!
	splice (@{$oldParent->[_C]}, $index, 1, ());

	$self->removeChildHoodMemories;
    }
    $self->[_Parent] = $parent;
}

#
# This function can return 3 values:
# 1: always readOnly
# 0: never readOnly
# undef: depends on parent node 
#
# Returns 1 for DocumentType, Notation, Entity, EntityReference, Attlist, 
# ElementDecl, AttDef. 
# The first 4 are readOnly according to the DOM Spec, the others are always 
# children of DocumentType. (Naturally, children of a readOnly node have to be
# readOnly as well...)
# These nodes are always readOnly regardless of who their ancestors are.
# Other nodes, e.g. Comment, are readOnly only if their parent is readOnly,
# which basically means that one of its ancestors has to be one of the
# aforementioned node types.
# Document and DocumentFragment return 0 for obvious reasons.
# Attr, Element, CDATASection, Text return 0. The DOM spec says that they can 
# be children of an Entity, but I don't think that that's possible
# with the current XML::Parser.
# Attr uses a {ReadOnly} property, which is only set if it's part of a AttDef.
# Always returns 0 if ignoreReadOnly is set.
#
sub isReadOnly
{
    # default implementation for Nodes that are always readOnly
    ! $XML::DOM::IgnoreReadOnly;
}

sub rejectChild
{
    1;
}

sub getNodeTypeName
{
    $NodeNames[$_[0]->getNodeType];
}

sub getChildIndex
{
    my ($self, $node) = @_;
    my $i = 0;

    return -1 unless defined $self->[_C];

    for my $kid (@{$self->[_C]})
    {
	return $i if $kid == $node;
	$i++;
    }
    -1;
}

sub getChildAtIndex
{
    my $kids = $_[0]->[_C];
    defined ($kids) ? $kids->[$_[1]] : undef;
}

sub isAncestor
{
    my ($self, $node) = @_;

    do
    {
	return 1 if $self == $node;
	$node = $node->[_Parent];
    }
    while (defined $node);

    0;
}

#
# Added for optimization. Overriden in XML::DOM::Text
#
sub isTextNode
{
    0;
}

#
# Added for optimization. Overriden in XML::DOM::DocumentFragment
#
sub isDocumentFragmentNode
{
    0;
}

#
# Added for optimization. Overriden in XML::DOM::Element
#
sub isElementNode
{
    0;
}

#
# Add a Text node with the specified value or append the text to the
# previous Node if it is a Text node.
#
sub addText
{
    # REC 9456 (if it was called)
    my ($self, $str) = @_;

    my $node = ${$self->[_C]}[-1];	# $self->getLastChild

    if (defined ($node) && $node->isTextNode)
    {
	# REC 5475 (if it was called)
	$node->appendData ($str);
    }
    else
    {
	$node = $self->[_Doc]->createTextNode ($str);
	$self->appendChild ($node);
    }
    $node;
}

#
# Add a CDATASection node with the specified value or append the text to the
# previous Node if it is a CDATASection node.
#
sub addCDATA
{
    my ($self, $str) = @_;

    my $node = ${$self->[_C]}[-1];	# $self->getLastChild

    if (defined ($node) && $node->getNodeType == CDATA_SECTION_NODE)
    {
	$node->appendData ($str);
    }
    else
    {
	$node = $self->[_Doc]->createCDATASection ($str);
	$self->appendChild ($node);
    }
}

sub removeChildNodes
{
    my $self = shift;

    my $cref = $self->[_C];
    return unless defined $cref;

    my $kid;
    while ($kid = pop @{$cref})
    {
	undef $kid->[_Parent]; # was delete
    }
}

sub toString
{
    my $self = shift;
    my $pr = $XML::DOM::PrintToString::Singleton;
    $pr->reset;
    $self->print ($pr);
    $pr->toString;
}

sub to_sax
{
    my $self = shift;
    unshift @_, 'Handler' if (@_ == 1);
    my %h = @_;

    my $doch = exists ($h{DocumentHandler}) ? $h{DocumentHandler} 
					    : $h{Handler};
    my $dtdh = exists ($h{DTDHandler}) ? $h{DTDHandler} 
				       : $h{Handler};
    my $enth = exists ($h{EntityResolver}) ? $h{EntityResolver} 
					   : $h{Handler};

    $self->_to_sax ($doch, $dtdh, $enth);
}

sub printToFile
{
    my ($self, $fileName) = @_;
    my $fh = new FileHandle ($fileName, "w") || 
	croak "printToFile - can't open output file $fileName";
    
    $self->print ($fh);
    $fh->close;
}

#
# Use print to print to a FileHandle object (see printToFile code)
#
sub printToFileHandle
{
    my ($self, $FH) = @_;
    my $pr = new XML::DOM::PrintToFileHandle ($FH);
    $self->print ($pr);
}

#
# Used by AttDef::setDefault to convert unexpanded default attribute value
#
sub expandEntityRefs
{
    my ($self, $str) = @_;
    my $doctype = $self->[_Doc]->getDoctype;

    use bytes;  # XML::RegExp expressed in terms encoded UTF8
    $str =~ s/&($XML::RegExp::Name|(#([0-9]+)|#x([0-9a-fA-F]+)));/
	defined($2) ? XML::DOM::XmlUtf8Encode ($3 || hex ($4)) 
		    : expandEntityRef ($1, $doctype)/ego;
    $str;
}

sub expandEntityRef
{
    my ($entity, $doctype) = @_;

    my $expanded = $XML::DOM::DefaultEntities{$entity};
    return $expanded if defined $expanded;

    $expanded = $doctype->getEntity ($entity);
    return $expanded->getValue if (defined $expanded);

#?? is this an error?
    croak "Could not expand entity reference of [$entity]\n";
#    return "&$entity;";	# entity not found
}

sub isHidden
{
    $_[0]->[_Hidden];
}

######################################################################
package XML::DOM::Attr;
######################################################################

use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Name Specified", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

sub new
{
    my ($class, $doc, $name, $value, $specified) = @_;

    if ($XML::DOM::SafeMode)
    {
	croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR,
					  "bad Attr name [$name]")
	    unless XML::DOM::isValidName ($name);
    }

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_C] = new XML::DOM::NodeList;
    $self->[_Name] = $name;
    
    if (defined $value)
    {
	$self->setValue ($value);
	$self->[_Specified] = (defined $specified) ? $specified : 1;
    }
    else
    {
	$self->[_Specified] = 0;
    }
    $self;
}

sub getNodeType
{
    ATTRIBUTE_NODE;
}

sub isSpecified
{
    $_[0]->[_Specified];
}

sub getName
{
    $_[0]->[_Name];
}

sub getValue
{
    my $self = shift;
    my $value = "";

    for my $kid (@{$self->[_C]})
    {
	$value .= $kid->getData if defined $kid->getData;
    }
    $value;
}

sub setValue
{
    my ($self, $value) = @_;

    # REC 1147
    $self->removeChildNodes;
    $self->appendChild ($self->[_Doc]->createTextNode ($value));
    $self->[_Specified] = 1;
}

sub getNodeName
{
    $_[0]->getName;
}

sub getNodeValue
{
    $_[0]->getValue;
}

sub setNodeValue
{
    $_[0]->setValue ($_[1]);
}

sub cloneNode
{
    my ($self) = @_;	# parameter deep is ignored

    my $node = $self->[_Doc]->createAttribute ($self->getName);
    $node->[_Specified] = $self->[_Specified];
    $node->[_ReadOnly] = 1 if $self->[_ReadOnly];

    $node->cloneChildren ($self, 1);
    $node;
}

#------------------------------------------------------------
# Extra method implementations
#

sub isReadOnly
{
    # ReadOnly property is set if it's part of a AttDef
    ! $XML::DOM::IgnoreReadOnly && defined ($_[0]->[_ReadOnly]);
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_Name];

    $FILE->print ("$name=\"");
    for my $kid (@{$self->[_C]})
    {
	if ($kid->getNodeType == TEXT_NODE)
	{
	    $FILE->print (XML::DOM::encodeAttrValue ($kid->getData));
	}
	else	# ENTITY_REFERENCE_NODE
	{
	    $kid->print ($FILE);
	}
    }
    $FILE->print ("\"");
}

sub rejectChild
{
    my $t = $_[1]->getNodeType;

    $t != TEXT_NODE 
    && $t != ENTITY_REFERENCE_NODE;
}

######################################################################
package XML::DOM::ProcessingInstruction;
######################################################################

use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };
BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Target Data", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

sub new
{
    my ($class, $doc, $target, $data, $hidden) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR,
			      "bad ProcessingInstruction Target [$target]")
	unless (XML::DOM::isValidName ($target) && $target !~ /^xml$/io);

    my $self = bless [], $class;
  
    $self->[_Doc] = $doc;
    $self->[_Target] = $target;
    $self->[_Data] = $data;
    $self->[_Hidden] = $hidden;
    $self;
}

sub getNodeType
{
    PROCESSING_INSTRUCTION_NODE;
}

sub getTarget
{
    $_[0]->[_Target];
}

sub getData
{
    $_[0]->[_Data];
}

sub setData
{
    my ($self, $data) = @_;

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    $self->[_Data] = $data;
}

sub getNodeName
{
    $_[0]->[_Target];
}

#
# Same as getData
#
sub getNodeValue
{
    $_[0]->[_Data];
}

sub setNodeValue
{
    $_[0]->setData ($_[1]);
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createProcessingInstruction ($self->getTarget, 
						$self->getData,
						$self->isHidden);
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    return 0 if $XML::DOM::IgnoreReadOnly;

    my $pa = $_[0]->[_Parent];
    defined ($pa) ? $pa->isReadOnly : 0;
}

sub print
{
    my ($self, $FILE) = @_;    

    $FILE->print ("<?");
    $FILE->print ($self->[_Target]);
    $FILE->print (" ");
    $FILE->print (XML::DOM::encodeProcessingInstruction ($self->[_Data]));
    $FILE->print ("?>");
}

sub _to_sax {
    my ($self, $doch) = @_;
    $doch->processing_instruction({Target => $self->getTarget, Data => $self->getData});
}

######################################################################
package XML::DOM::Notation;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Name Base SysId PubId", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

sub new
{
    my ($class, $doc, $name, $base, $sysId, $pubId, $hidden) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
				      "bad Notation Name [$name]")
	unless XML::DOM::isValidName ($name);

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_Name] = $name;
    $self->[_Base] = $base;
    $self->[_SysId] = $sysId;
    $self->[_PubId] = $pubId;
    $self->[_Hidden] = $hidden;
    $self;
}

sub getNodeType
{
    NOTATION_NODE;
}

sub getPubId
{
    $_[0]->[_PubId];
}

sub setPubId
{
    $_[0]->[_PubId] = $_[1];
}

sub getSysId
{
    $_[0]->[_SysId];
}

sub setSysId
{
    $_[0]->[_SysId] = $_[1];
}

sub getName
{
    $_[0]->[_Name];
}

sub setName
{
    $_[0]->[_Name] = $_[1];
}

sub getBase
{
    $_[0]->[_Base];
}

sub getNodeName
{
    $_[0]->[_Name];
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_Name];
    my $sysId = $self->[_SysId];
    my $pubId = $self->[_PubId];

    $FILE->print ("<!NOTATION $name ");

    if (defined $pubId)
    {
	$FILE->print (" PUBLIC \"$pubId\"");	
    }
    if (defined $sysId)
    {
	$FILE->print (" SYSTEM \"$sysId\"");	
    }
    $FILE->print (">");
}

sub cloneNode
{
    my ($self) = @_;
    $self->[_Doc]->createNotation ($self->[_Name], $self->[_Base], 
				   $self->[_SysId], $self->[_PubId],
				   $self->[_Hidden]);
}

sub to_expat
{
    my ($self, $iter) = @_;
    $iter->Notation ($self->getName, $self->getBase, 
		     $self->getSysId, $self->getPubId);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    $dtdh->notation_decl ( { Name => $self->getName, 
			     Base => $self->getBase, 
			     SystemId => $self->getSysId, 
			     PublicId => $self->getPubId });
}

######################################################################
package XML::DOM::Entity;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("NotationName Parameter Value Ndata SysId PubId", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

sub new
{
    my ($class, $doc, $notationName, $value, $sysId, $pubId, $ndata, $isParam, $hidden) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
				      "bad Entity Name [$notationName]")
	unless XML::DOM::isValidName ($notationName);

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_NotationName] = $notationName;
    $self->[_Parameter] = $isParam;
    $self->[_Value] = $value;
    $self->[_Ndata] = $ndata;
    $self->[_SysId] = $sysId;
    $self->[_PubId] = $pubId;
    $self->[_Hidden] = $hidden;
    $self;
#?? maybe Value should be a Text node
}

sub getNodeType
{
    ENTITY_NODE;
}

sub getPubId
{
    $_[0]->[_PubId];
}

sub getSysId
{
    $_[0]->[_SysId];
}

# Dom Spec says: 
#  For unparsed entities, the name of the notation for the
#  entity. For parsed entities, this is null.

#?? do we have unparsed entities?
sub getNotationName
{
    $_[0]->[_NotationName];
}

sub getNodeName
{
    $_[0]->[_NotationName];
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createEntity ($self->[_NotationName], $self->[_Value], 
				 $self->[_SysId], $self->[_PubId], 
				 $self->[_Ndata], $self->[_Parameter], $self->[_Hidden]);
}

sub rejectChild
{
    return 1;
#?? if value is split over subnodes, recode this section
# also add:				   C => new XML::DOM::NodeList,

    my $t = $_[1];

    return $t == TEXT_NODE
	|| $t == ENTITY_REFERENCE_NODE 
	|| $t == PROCESSING_INSTRUCTION_NODE
	|| $t == COMMENT_NODE
	|| $t == CDATA_SECTION_NODE
	|| $t == ELEMENT_NODE;
}

sub getValue
{
    $_[0]->[_Value];
}

sub isParameterEntity
{
    $_[0]->[_Parameter];
}

sub getNdata
{
    $_[0]->[_Ndata];
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_NotationName];

    my $par = $self->isParameterEntity ? "% " : "";

    $FILE->print ("<!ENTITY $par$name");

    my $value = $self->[_Value];
    my $sysId = $self->[_SysId];
    my $pubId = $self->[_PubId];
    my $ndata = $self->[_Ndata];

    if (defined $value)
    {
#?? Not sure what to do if it contains both single and double quote
	$value = ($value =~ /\"/) ? "'$value'" : "\"$value\"";
	$FILE->print (" $value");
    }
    if (defined $pubId)
    {
	$FILE->print (" PUBLIC \"$pubId\"");	
    }
    elsif (defined $sysId)
    {
	$FILE->print (" SYSTEM");
    }

    if (defined $sysId)
    {
	$FILE->print (" \"$sysId\"");
    }
    $FILE->print (" NDATA $ndata") if defined $ndata;
    $FILE->print (">");
}

sub to_expat
{
    my ($self, $iter) = @_;
    my $name = ($self->isParameterEntity ? '%' : "") . $self->getNotationName; 
    $iter->Entity ($name,
		   $self->getValue, $self->getSysId, $self->getPubId, 
		   $self->getNdata);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    my $name = ($self->isParameterEntity ? '%' : "") . $self->getNotationName; 
    $dtdh->entity_decl ( { Name => $name, 
			   Value => $self->getValue, 
			   SystemId => $self->getSysId, 
			   PublicId => $self->getPubId, 
			   Notation => $self->getNdata } );
}

######################################################################
package XML::DOM::EntityReference;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("EntityName Parameter NoExpand", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

sub new
{
    my ($class, $doc, $name, $parameter, $noExpand) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
		      "bad Entity Name [$name] in EntityReference")
	unless XML::DOM::isValidName ($name);

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_EntityName] = $name;
    $self->[_Parameter] = ($parameter || 0);
    $self->[_NoExpand] = ($noExpand || 0);

    $self;
}

sub getNodeType
{
    ENTITY_REFERENCE_NODE;
}

sub getNodeName
{
    $_[0]->[_EntityName];
}

#------------------------------------------------------------
# Extra method implementations

sub getEntityName
{
    $_[0]->[_EntityName];
}

sub isParameterEntity
{
    $_[0]->[_Parameter];
}

sub getData
{
    my $self = shift;
    my $name = $self->[_EntityName];
    my $parameter = $self->[_Parameter];

    my $data;
    if ($self->[_NoExpand]) {
      $data = "&$name;" if $name;
    } else {
      $data = $self->[_Doc]->expandEntity ($name, $parameter);
    }

    unless (defined $data)
    {
#?? this is probably an error, but perhaps requires check to NoExpand
# will fix it?
	my $pc = $parameter ? "%" : "&";
	$data = "$pc$name;";
    }
    $data;
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_EntityName];

#?? or do we expand the entities?

    my $pc = $self->[_Parameter] ? "%" : "&";
    $FILE->print ("$pc$name;");
}

# Dom Spec says:
#     [...] but if such an Entity exists, then
#     the child list of the EntityReference node is the same as that of the
#     Entity node. 
#
#     The resolution of the children of the EntityReference (the replacement
#     value of the referenced Entity) may be lazily evaluated; actions by the
#     user (such as calling the childNodes method on the EntityReference
#     node) are assumed to trigger the evaluation.
sub getChildNodes
{
    my $self = shift;
    my $entity = $self->[_Doc]->getEntity ($self->[_EntityName]);
    defined ($entity) ? $entity->getChildNodes : new XML::DOM::NodeList;
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createEntityReference ($self->[_EntityName], 
                                         $self->[_Parameter],
                                         $self->[_NoExpand],
                                          );
}

sub to_expat
{
    my ($self, $iter) = @_;
    $iter->EntityRef ($self->getEntityName, $self->isParameterEntity);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    my @par = $self->isParameterEntity ? (Parameter => 1) : ();
#?? not supported by PerlSAX: $self->isParameterEntity

    $doch->entity_reference ( { Name => $self->getEntityName, @par } );
}

# NOTE: an EntityReference can't really have children, so rejectChild
# is not reimplemented (i.e. it always returns 0.)

######################################################################
package XML::DOM::AttDef;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Name Type Fixed Default Required Implied Quote", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

#------------------------------------------------------------
# Extra method implementations

# AttDef is not part of DOM Spec
sub new
{
    my ($class, $doc, $name, $attrType, $default, $fixed, $hidden) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR,
				      "bad Attr name in AttDef [$name]")
	unless XML::DOM::isValidName ($name);

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_Name] = $name;
    $self->[_Type] = $attrType;

    if (defined $default)
    {
	if ($default eq "#REQUIRED")
	{
	    $self->[_Required] = 1;
	}
	elsif ($default eq "#IMPLIED")
	{
	    $self->[_Implied] = 1;
	}
	else
	{
	    # strip off quotes - see Attlist handler in XML::Parser
            # this regexp doesn't work with 5.8.0 unicode
#	    $default =~ m#^(["'])(.*)['"]$#;
#	    $self->[_Quote] = $1;	# keep track of the quote character
#	    $self->[_Default] = $self->setDefault ($2);

          # workaround for 5.8.0 unicode
          $default =~ s!^(["'])!!;
          $self->[_Quote] = $1;
          $default =~ s!(["'])$!!;
          $self->[_Default] = $self->setDefault ($default);
	    	    
#?? should default value be decoded - what if it contains e.g. "&amp;"
	}
    }
    $self->[_Fixed] = $fixed if defined $fixed;
    $self->[_Hidden] = $hidden if defined $hidden;

    $self;
}

sub getNodeType
{
    ATT_DEF_NODE;
}

sub getName
{
    $_[0]->[_Name];
}

# So it can be added to a NamedNodeMap
sub getNodeName
{
    $_[0]->[_Name];
}

sub getType
{
    $_[0]->[_Type];
}

sub setType
{
    $_[0]->[_Type] = $_[1];
}

sub getDefault
{
    $_[0]->[_Default];
}

sub setDefault
{
    my ($self, $value) = @_;

    # specified=0, it's the default !
    my $attr = $self->[_Doc]->createAttribute ($self->[_Name], undef, 0);
    $attr->[_ReadOnly] = 1;

#?? this should be split over Text and EntityReference nodes, just like other
# Attr nodes - just expand the text for now
    $value = $self->expandEntityRefs ($value);
    $attr->addText ($value);
#?? reimplement in NoExpand mode!

    $attr;
}

sub isFixed
{
    $_[0]->[_Fixed] || 0;
}

sub isRequired
{
    $_[0]->[_Required] || 0;
}

sub isImplied
{
    $_[0]->[_Implied] || 0;
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_Name];
    my $type = $self->[_Type];
    my $fixed = $self->[_Fixed];
    my $default = $self->[_Default];

#    $FILE->print ("$name $type");
    # replaced line above with the two lines below
    # seems to be a bug in perl 5.6.0 that causes
    # test 3 of dom_jp_attr.t to fail?
    $FILE->print ($name);
    $FILE->print (" $type");

    $FILE->print (" #FIXED") if defined $fixed;

    if ($self->[_Required])
    {
	$FILE->print (" #REQUIRED");
    }
    elsif ($self->[_Implied])
    {
	$FILE->print (" #IMPLIED");
    }
    elsif (defined ($default))
    {
	my $quote = $self->[_Quote];
	$FILE->print (" $quote");
	for my $kid (@{$default->[_C]})
	{
	    $kid->print ($FILE);
	}
	$FILE->print ($quote);	
    }
}

sub getDefaultString
{
    my $self = shift;
    my $default;

    if ($self->[_Required])
    {
	return "#REQUIRED";
    }
    elsif ($self->[_Implied])
    {
	return "#IMPLIED";
    }
    elsif (defined ($default = $self->[_Default]))
    {
	my $quote = $self->[_Quote];
	$default = $default->toString;
	return "$quote$default$quote";
    }
    undef;
}

sub cloneNode
{
    my $self = shift;
    my $node = new XML::DOM::AttDef ($self->[_Doc], $self->[_Name], $self->[_Type],
				     undef, $self->[_Fixed]);

    $node->[_Required] = 1 if $self->[_Required];
    $node->[_Implied] = 1 if $self->[_Implied];
    $node->[_Fixed] = $self->[_Fixed] if defined $self->[_Fixed];
    $node->[_Hidden] = $self->[_Hidden] if defined $self->[_Hidden];

    if (defined $self->[_Default])
    {
	$node->[_Default] = $self->[_Default]->cloneNode(1);
    }
    $node->[_Quote] = $self->[_Quote];

    $node;
}

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    $self->SUPER::setOwnerDocument ($doc);

    if (defined $self->[_Default])
    {
	$self->[_Default]->setOwnerDocument ($doc);
    }
}

######################################################################
package XML::DOM::AttlistDecl;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    import XML::DOM::AttDef qw{ :Fields };

    XML::DOM::def_fields ("ElementName", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;

#------------------------------------------------------------
# Extra method implementations

# AttlistDecl is not part of the DOM Spec
sub new
{
    my ($class, $doc, $name) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
			      "bad Element TagName [$name] in AttlistDecl")
	unless XML::DOM::isValidName ($name);

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_C] = new XML::DOM::NodeList;
    $self->[_ReadOnly] = 1;
    $self->[_ElementName] = $name;

    $self->[_A] = new XML::DOM::NamedNodeMap (Doc	=> $doc,
					      ReadOnly	=> 1,
					      Parent	=> $self);

    $self;
}

sub getNodeType
{
    ATTLIST_DECL_NODE;
}

sub getName
{
    $_[0]->[_ElementName];
}

sub getNodeName
{
    $_[0]->[_ElementName];
}

sub getAttDef
{
    my ($self, $attrName) = @_;
    $self->[_A]->getNamedItem ($attrName);
}

sub addAttDef
{
    my ($self, $attrName, $type, $default, $fixed, $hidden) = @_;
    my $node = $self->getAttDef ($attrName);

    if (defined $node)
    {
	# data will be ignored if already defined
	my $elemName = $self->getName;
	XML::DOM::warning ("multiple definitions of attribute $attrName for element $elemName, only first one is recognized");
    }
    else
    {
	$node = new XML::DOM::AttDef ($self->[_Doc], $attrName, $type, 
				      $default, $fixed, $hidden);
	$self->[_A]->setNamedItem ($node);
    }
    $node;
}

sub getDefaultAttrValue
{
    my ($self, $attr) = @_;
    my $attrNode = $self->getAttDef ($attr);
    (defined $attrNode) ? $attrNode->getDefault : undef;
}

sub cloneNode
{
    my ($self, $deep) = @_;
    my $node = $self->[_Doc]->createAttlistDecl ($self->[_ElementName]);
    
    $node->[_A] = $self->[_A]->cloneNode ($deep);
    $node;
}

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    $self->SUPER::setOwnerDocument ($doc);

    $self->[_A]->setOwnerDocument ($doc);
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->getName;
    my @attlist = @{$self->[_A]->getValues};

    my $hidden = 1;
    for my $att (@attlist)
    {
	unless ($att->[_Hidden])
	{
	    $hidden = 0;
	    last;
	}
    }

    unless ($hidden)
    {
	$FILE->print ("<!ATTLIST $name");

	if (@attlist == 1)
	{
	    $FILE->print (" ");
	    $attlist[0]->print ($FILE);	    
	}
	else
	{
	    for my $attr (@attlist)
	    {
		next if $attr->[_Hidden];

		$FILE->print ("\x0A  ");
		$attr->print ($FILE);
	    }
	}
	$FILE->print (">");
    }
}

sub to_expat
{
    my ($self, $iter) = @_;
    my $tag = $self->getName;
    for my $a ($self->[_A]->getValues)
    {
	my $default = $a->isImplied ? '#IMPLIED' :
	    ($a->isRequired ? '#REQUIRED' : 
	     ($a->[_Quote] . $a->getDefault->getValue . $a->[_Quote]));

	$iter->Attlist ($tag, $a->getName, $a->getType, $default, $a->isFixed); 
    }
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    my $tag = $self->getName;
    for my $a ($self->[_A]->getValues)
    {
	my $default = $a->isImplied ? '#IMPLIED' :
	    ($a->isRequired ? '#REQUIRED' : 
	     ($a->[_Quote] . $a->getDefault->getValue . $a->[_Quote]));

	$dtdh->attlist_decl ({ ElementName => $tag, 
			       AttributeName => $a->getName, 
			       Type => $a->[_Type], 
			       Default => $default, 
			       Fixed => $a->isFixed }); 
    }
}

######################################################################
package XML::DOM::ElementDecl;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Name Model", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;


#------------------------------------------------------------
# Extra method implementations

# ElementDecl is not part of the DOM Spec
sub new
{
    my ($class, $doc, $name, $model, $hidden) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
			      "bad Element TagName [$name] in ElementDecl")
	unless XML::DOM::isValidName ($name);

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_Name] = $name;
    $self->[_ReadOnly] = 1;
    $self->[_Model] = $model;
    $self->[_Hidden] = $hidden;
    $self;
}

sub getNodeType
{
    ELEMENT_DECL_NODE;
}

sub getName
{
    $_[0]->[_Name];
}

sub getNodeName
{
    $_[0]->[_Name];
}

sub getModel
{
    $_[0]->[_Model];
}

sub setModel
{
    my ($self, $model) = @_;

    $self->[_Model] = $model;
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_Name];
    my $model = $self->[_Model];

    $FILE->print ("<!ELEMENT $name $model>")
	unless $self->[_Hidden];
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createElementDecl ($self->[_Name], $self->[_Model], 
				      $self->[_Hidden]);
}

sub to_expat
{
#?? add support for Hidden?? (allover, also in _to_sax!!)

    my ($self, $iter) = @_;
    $iter->Element ($self->getName, $self->getModel);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    $dtdh->element_decl ( { Name => $self->getName, 
			    Model => $self->getModel } );
}

######################################################################
package XML::DOM::Element;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("TagName", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use XML::DOM::NamedNodeMap;
use Carp;

sub new
{
    my ($class, $doc, $tagName) = @_;

    if ($XML::DOM::SafeMode)
    {
	croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
				      "bad Element TagName [$tagName]")
	    unless XML::DOM::isValidName ($tagName);
    }

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_C] = new XML::DOM::NodeList;
    $self->[_TagName] = $tagName;

# Now we're creating the NamedNodeMap only when needed (REC 2313 => 1147)    
#    $self->[_A] = new XML::DOM::NamedNodeMap (Doc	=> $doc,
#					     Parent	=> $self);

    $self;
}

sub getNodeType
{
    ELEMENT_NODE;
}

sub getTagName
{
    $_[0]->[_TagName];
}

sub getNodeName
{
    $_[0]->[_TagName];
}

sub getAttributeNode
{
    my ($self, $name) = @_;
    return undef unless defined $self->[_A];

    $self->getAttributes->{$name};
}

sub getAttribute
{
    my ($self, $name) = @_;
    my $attr = $self->getAttributeNode ($name);
    (defined $attr) ? $attr->getValue : "";
}

sub setAttribute
{
    my ($self, $name, $val) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR,
				      "bad Attr Name [$name]")
	unless XML::DOM::isValidName ($name);

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    my $node = $self->getAttributes->{$name};
    if (defined $node)
    {
	$node->setValue ($val);
    }
    else
    {
	$node = $self->[_Doc]->createAttribute ($name, $val);
	$self->[_A]->setNamedItem ($node);
    }
}

sub setAttributeNode
{
    my ($self, $node) = @_;
    my $attr = $self->getAttributes;
    my $name = $node->getNodeName;

    # REC 1147
    if ($XML::DOM::SafeMode)
    {
	croak new XML::DOM::DOMException (WRONG_DOCUMENT_ERR,
					  "nodes belong to different documents")
	    if $self->[_Doc] != $node->[_Doc];

	croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
					  "node is ReadOnly")
	    if $self->isReadOnly;

	my $attrParent = $node->[_UsedIn];
	croak new XML::DOM::DOMException (INUSE_ATTRIBUTE_ERR,
					  "Attr is already used by another Element")
	    if (defined ($attrParent) && $attrParent != $attr);
    }

    my $other = $attr->{$name};
    $attr->removeNamedItem ($name) if defined $other;

    $attr->setNamedItem ($node);

    $other;
}

sub removeAttributeNode
{
    my ($self, $node) = @_;

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    my $attr = $self->[_A];
    unless (defined $attr)
    {
	croak new XML::DOM::DOMException (NOT_FOUND_ERR);
	return undef;
    }

    my $name = $node->getNodeName;
    my $attrNode = $attr->getNamedItem ($name);

#?? should it croak if it's the default value?
    croak new XML::DOM::DOMException (NOT_FOUND_ERR)
	unless $node == $attrNode;

    # Not removing anything if it's the default value already
    return undef unless $node->isSpecified;

    $attr->removeNamedItem ($name);

    # Substitute with default value if it's defined
    my $default = $self->getDefaultAttrValue ($name);
    if (defined $default)
    {
	local $XML::DOM::IgnoreReadOnly = 1;

	$default = $default->cloneNode (1);
	$attr->setNamedItem ($default);
    }
    $node;
}

sub removeAttribute
{
    my ($self, $name) = @_;
    my $attr = $self->[_A];
    unless (defined $attr)
    {
	croak new XML::DOM::DOMException (NOT_FOUND_ERR);
	return;
    }
    
    my $node = $attr->getNamedItem ($name);
    if (defined $node)
    {
#?? could use dispose() to remove circular references for gc, but what if
#?? somebody is referencing it?
	$self->removeAttributeNode ($node);
    }
}

sub cloneNode
{
    my ($self, $deep) = @_;
    my $node = $self->[_Doc]->createElement ($self->getTagName);

    # Always clone the Attr nodes, even if $deep == 0
    if (defined $self->[_A])
    {
	$node->[_A] = $self->[_A]->cloneNode (1);	# deep=1
	$node->[_A]->setParentNode ($node);
    }

    $node->cloneChildren ($self, $deep);
    $node;
}

sub getAttributes
{
    $_[0]->[_A] ||= XML::DOM::NamedNodeMap->new (Doc	=> $_[0]->[_Doc],
						 Parent	=> $_[0]);
}

#------------------------------------------------------------
# Extra method implementations

# Added for convenience
sub setTagName
{
    my ($self, $tagName) = @_;

    croak new XML::DOM::DOMException (INVALID_CHARACTER_ERR, 
				      "bad Element TagName [$tagName]")
        unless XML::DOM::isValidName ($tagName);

    $self->[_TagName] = $tagName;
}

sub isReadOnly
{
    0;
}

# Added for optimization.
sub isElementNode
{
    1;
}

sub rejectChild
{
    my $t = $_[1]->getNodeType;

    $t != TEXT_NODE
    && $t != ENTITY_REFERENCE_NODE 
    && $t != PROCESSING_INSTRUCTION_NODE
    && $t != COMMENT_NODE
    && $t != CDATA_SECTION_NODE
    && $t != ELEMENT_NODE;
}

sub getDefaultAttrValue
{
    my ($self, $attr) = @_;
    $self->[_Doc]->getDefaultAttrValue ($self->[_TagName], $attr);
}

sub dispose
{
    my $self = shift;

    $self->[_A]->dispose if defined $self->[_A];
    $self->SUPER::dispose;
}

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    $self->SUPER::setOwnerDocument ($doc);

    $self->[_A]->setOwnerDocument ($doc) if defined $self->[_A];
}

sub print
{
    my ($self, $FILE) = @_;    

    my $name = $self->[_TagName];

    $FILE->print ("<$name");

    if (defined $self->[_A])
    {
	for my $att (@{$self->[_A]->getValues})
	{
	    # skip un-specified (default) Attr nodes
	    if ($att->isSpecified)
	    {
		$FILE->print (" ");
		$att->print ($FILE);
	    }
	}
    }

    my @kids = @{$self->[_C]};
    if (@kids > 0)
    {
	$FILE->print (">");
	for my $kid (@kids)
	{
	    $kid->print ($FILE);
	}
	$FILE->print ("</$name>");
    }
    else
    {
	my $style = &$XML::DOM::TagStyle ($name, $self);
	if ($style == 0)
	{
	    $FILE->print ("/>");
	}
	elsif ($style == 1)
	{
	    $FILE->print ("></$name>");
	}
	else
	{
	    $FILE->print (" />");
	}
    }
}

sub check
{
    my ($self, $checker) = @_;
    die "Usage: \$xml_dom_elem->check (\$checker)" unless $checker; 

    $checker->InitDomElem;
    $self->to_expat ($checker);
    $checker->FinalDomElem;
}

sub to_expat
{
    my ($self, $iter) = @_;

    my $tag = $self->getTagName;
    $iter->Start ($tag);

    if (defined $self->[_A])
    {
	for my $attr ($self->[_A]->getValues)
	{
	    $iter->Attr ($tag, $attr->getName, $attr->getValue, $attr->isSpecified);
	}
    }

    $iter->EndAttr;

    for my $kid ($self->getChildNodes)
    {
	$kid->to_expat ($iter);
    }

    $iter->End;
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;

    my $tag = $self->getTagName;

    my @attr = ();
    my $attrOrder;
    my $attrDefaulted;

    if (defined $self->[_A])
    {
	my @spec = ();		# names of specified attributes
	my @unspec = ();	# names of defaulted attributes

	for my $attr ($self->[_A]->getValues) 
	{
	    my $attrName = $attr->getName;
	    push @attr, $attrName, $attr->getValue;
	    if ($attr->isSpecified)
	    {
		push @spec, $attrName;
	    }
	    else
	    {
		push @unspec, $attrName;
	    }
	}
	$attrOrder = [ @spec, @unspec ];
	$attrDefaulted = @spec;
    }
    $doch->start_element (defined $attrOrder ? 
			  { Name => $tag, 
			    Attributes => { @attr },
			    AttributeOrder => $attrOrder,
			    Defaulted => $attrDefaulted
			  } :
			  { Name => $tag, 
			    Attributes => { @attr } 
			  }
			 );

    for my $kid ($self->getChildNodes)
    {
	$kid->_to_sax ($doch, $dtdh, $enth);
    }

    $doch->end_element ( { Name => $tag } );
}

######################################################################
package XML::DOM::CharacterData;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Data", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use Carp;


#
# CharacterData nodes should never be created directly, only subclassed!
#
sub new
{
    my ($class, $doc, $data) = @_;
    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_Data] = $data;
    $self;
}

sub appendData
{
    my ($self, $data) = @_;

    if ($XML::DOM::SafeMode)
    {
	croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
					  "node is ReadOnly")
	    if $self->isReadOnly;
    }
    $self->[_Data] .= $data;
}

sub deleteData
{
    my ($self, $offset, $count) = @_;

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "bad offset [$offset]")
	if ($offset < 0 || $offset >= length ($self->[_Data]));
#?? DOM Spec says >, but >= makes more sense!

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "negative count [$count]")
	if $count < 0;
 
    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    substr ($self->[_Data], $offset, $count) = "";
}

sub getData
{
    $_[0]->[_Data];
}

sub getLength
{
    length $_[0]->[_Data];
}

sub insertData
{
    my ($self, $offset, $data) = @_;

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "bad offset [$offset]")
	if ($offset < 0 || $offset >= length ($self->[_Data]));
#?? DOM Spec says >, but >= makes more sense!

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    substr ($self->[_Data], $offset, 0) = $data;
}

sub replaceData
{
    my ($self, $offset, $count, $data) = @_;

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "bad offset [$offset]")
	if ($offset < 0 || $offset >= length ($self->[_Data]));
#?? DOM Spec says >, but >= makes more sense!

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "negative count [$count]")
	if $count < 0;
 
    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    substr ($self->[_Data], $offset, $count) = $data;
}

sub setData
{
    my ($self, $data) = @_;

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    $self->[_Data] = $data;
}

sub substringData
{
    my ($self, $offset, $count) = @_;
    my $data = $self->[_Data];

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "bad offset [$offset]")
	if ($offset < 0 || $offset >= length ($data));
#?? DOM Spec says >, but >= makes more sense!

    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "negative count [$count]")
	if $count < 0;
    
    substr ($data, $offset, $count);
}

sub getNodeValue
{
    $_[0]->getData;
}

sub setNodeValue
{
    $_[0]->setData ($_[1]);
}

######################################################################
package XML::DOM::CDATASection;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::CharacterData qw( :DEFAULT :Fields );
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("", "XML::DOM::CharacterData");
}

use XML::DOM::DOMException;

sub getNodeName
{
    "#cdata-section";
}

sub getNodeType
{
    CDATA_SECTION_NODE;
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createCDATASection ($self->getData);
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    0;
}

sub print
{
    my ($self, $FILE) = @_;
    $FILE->print ("<![CDATA[");
    $FILE->print (XML::DOM::encodeCDATA ($self->getData));
    $FILE->print ("]]>");
}

sub to_expat
{
    my ($self, $iter) = @_;
    $iter->CData ($self->getData);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    $doch->start_cdata;
    $doch->characters ( { Data => $self->getData } );
    $doch->end_cdata;
}

######################################################################
package XML::DOM::Comment;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::CharacterData qw( :DEFAULT :Fields );
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("", "XML::DOM::CharacterData");
}

use XML::DOM::DOMException;
use Carp;

#?? setData - could check comment for double minus

sub getNodeType
{
    COMMENT_NODE;
}

sub getNodeName
{
    "#comment";
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createComment ($self->getData);
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    return 0 if $XML::DOM::IgnoreReadOnly;

    my $pa = $_[0]->[_Parent];
    defined ($pa) ? $pa->isReadOnly : 0;
}

sub print
{
    my ($self, $FILE) = @_;
    my $comment = XML::DOM::encodeComment ($self->[_Data]);

    $FILE->print ("<!--$comment-->");
}

sub to_expat
{
    my ($self, $iter) = @_;
    $iter->Comment ($self->getData);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    $doch->comment ( { Data => $self->getData });
}

######################################################################
package XML::DOM::Text;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::CharacterData qw( :DEFAULT :Fields );
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("", "XML::DOM::CharacterData");
}

use XML::DOM::DOMException;
use Carp;

sub getNodeType
{
    TEXT_NODE;
}

sub getNodeName
{
    "#text";
}

sub splitText
{
    my ($self, $offset) = @_;

    my $data = $self->getData;
    croak new XML::DOM::DOMException (INDEX_SIZE_ERR,
				      "bad offset [$offset]")
	if ($offset < 0 || $offset >= length ($data));
#?? DOM Spec says >, but >= makes more sense!

    croak new XML::DOM::DOMException (NO_MODIFICATION_ALLOWED_ERR,
				      "node is ReadOnly")
	if $self->isReadOnly;

    my $rest = substr ($data, $offset);

    $self->setData (substr ($data, 0, $offset));
    my $node = $self->[_Doc]->createTextNode ($rest);

    # insert new node after this node
    $self->[_Parent]->insertBefore ($node, $self->getNextSibling);

    $node;
}

sub cloneNode
{
    my $self = shift;
    $self->[_Doc]->createTextNode ($self->getData);
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    0;
}

sub print
{
    my ($self, $FILE) = @_;
    $FILE->print (XML::DOM::encodeText ($self->getData, '<&>"'));
}

sub isTextNode
{
    1;
}

sub to_expat
{
    my ($self, $iter) = @_;
    $iter->Char ($self->getData);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    $doch->characters ( { Data => $self->getData } );
}

######################################################################
package XML::DOM::XMLDecl;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Version Encoding Standalone", "XML::DOM::Node");
}

use XML::DOM::DOMException;


#------------------------------------------------------------
# Extra method implementations

# XMLDecl is not part of the DOM Spec
sub new
{
    my ($class, $doc, $version, $encoding, $standalone) = @_;

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_Version] = $version if defined $version;
    $self->[_Encoding] = $encoding if defined $encoding;
    $self->[_Standalone] = $standalone if defined $standalone;

    $self;
}

sub setVersion
{
    if (defined $_[1])
    {
	$_[0]->[_Version] = $_[1];
    }
    else
    {
	undef $_[0]->[_Version]; # was delete
    }
}

sub getVersion
{
    $_[0]->[_Version];
}

sub setEncoding
{
    if (defined $_[1])
    {
	$_[0]->[_Encoding] = $_[1];
    }
    else
    {
	undef $_[0]->[_Encoding]; # was delete
    }
}

sub getEncoding
{
    $_[0]->[_Encoding];
}

sub setStandalone
{
    if (defined $_[1])
    {
	$_[0]->[_Standalone] = $_[1];
    }
    else
    {
	undef $_[0]->[_Standalone]; # was delete
    }
}

sub getStandalone
{
    $_[0]->[_Standalone];
}

sub getNodeType
{
    XML_DECL_NODE;
}

sub cloneNode
{
    my $self = shift;

    new XML::DOM::XMLDecl ($self->[_Doc], $self->[_Version], 
			   $self->[_Encoding], $self->[_Standalone]);
}

sub print
{
    my ($self, $FILE) = @_;

    my $version = $self->[_Version];
    my $encoding = $self->[_Encoding];
    my $standalone = $self->[_Standalone];
    $standalone = ($standalone ? "yes" : "no") if defined $standalone;

    $FILE->print ("<?xml");
    $FILE->print (" version=\"$version\"")	 if defined $version;    
    $FILE->print (" encoding=\"$encoding\"")	 if defined $encoding;
    $FILE->print (" standalone=\"$standalone\"") if defined $standalone;
    $FILE->print ("?>");
}

sub to_expat
{
    my ($self, $iter) = @_;
    $iter->XMLDecl ($self->getVersion, $self->getEncoding, $self->getStandalone);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;
    $dtdh->xml_decl ( { Version => $self->getVersion, 
			Encoding => $self->getEncoding, 
			Standalone => $self->getStandalone } );
}

######################################################################
package XML::DOM::DocumentFragment;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("", "XML::DOM::Node");
}

use XML::DOM::DOMException;

sub new
{
    my ($class, $doc) = @_;
    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_C] = new XML::DOM::NodeList;
    $self;
}

sub getNodeType
{
    DOCUMENT_FRAGMENT_NODE;
}

sub getNodeName
{
    "#document-fragment";
}

sub cloneNode
{
    my ($self, $deep) = @_;
    my $node = $self->[_Doc]->createDocumentFragment;

    $node->cloneChildren ($self, $deep);
    $node;
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    0;
}

sub print
{
    my ($self, $FILE) = @_;

    for my $node (@{$self->[_C]})
    {
	$node->print ($FILE);
    }
}

sub rejectChild
{
    my $t = $_[1]->getNodeType;

    $t != TEXT_NODE
	&& $t != ENTITY_REFERENCE_NODE 
	&& $t != PROCESSING_INSTRUCTION_NODE
	&& $t != COMMENT_NODE
	&& $t != CDATA_SECTION_NODE
	&& $t != ELEMENT_NODE;
}

sub isDocumentFragmentNode
{
    1;
}

######################################################################
package XML::DOM::DocumentType;		# forward declaration
######################################################################

######################################################################
package XML::DOM::Document;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    XML::DOM::def_fields ("Doctype XmlDecl", "XML::DOM::Node");
}

use Carp;
use XML::DOM::NodeList;
use XML::DOM::DOMException;

sub new
{
    my ($class) = @_;
    my $self = bless [], $class;

    # keep Doc pointer, even though getOwnerDocument returns undef
    $self->[_Doc] = $self;
    $self->[_C] = new XML::DOM::NodeList;
    $self;
}

sub getNodeType
{
    DOCUMENT_NODE;
}

sub getNodeName
{
    "#document";
}

#?? not sure about keeping a fixed order of these nodes....
sub getDoctype
{
    $_[0]->[_Doctype];
}

sub getDocumentElement
{
    my ($self) = @_;
    for my $kid (@{$self->[_C]})
    {
	return $kid if $kid->isElementNode;
    }
    undef;
}

sub getOwnerDocument
{
    undef;
}

sub getImplementation 
{
    $XML::DOM::DOMImplementation::Singleton;
}

#
# Added extra parameters ($val, $specified) that are passed straight to the
# Attr constructor
# 
sub createAttribute
{
    new XML::DOM::Attr (@_);
}

sub createCDATASection
{
    new XML::DOM::CDATASection (@_);
}

sub createComment
{
    new XML::DOM::Comment (@_);

}

sub createElement
{
    new XML::DOM::Element (@_);
}

sub createTextNode
{
    new XML::DOM::Text (@_);
}

sub createProcessingInstruction
{
    new XML::DOM::ProcessingInstruction (@_);
}

sub createEntityReference
{
    new XML::DOM::EntityReference (@_);
}

sub createDocumentFragment
{
    new XML::DOM::DocumentFragment (@_);
}

sub createDocumentType
{
    new XML::DOM::DocumentType (@_);
}

sub cloneNode
{
    my ($self, $deep) = @_;
    my $node = new XML::DOM::Document;

    $node->cloneChildren ($self, $deep);

    my $xmlDecl = $self->[_XmlDecl];
    $node->[_XmlDecl] = $xmlDecl->cloneNode ($deep) if defined $xmlDecl;

    $node;
}

sub appendChild
{
    my ($self, $node) = @_;

    # Extra check: make sure we don't end up with more than one Element.
    # Don't worry about multiple DocType nodes, because DocumentFragment
    # can't contain DocType nodes.

    my @nodes = ($node);
    @nodes = @{$node->[_C]}
        if $node->getNodeType == DOCUMENT_FRAGMENT_NODE;
    
    my $elem = 0;
    for my $n (@nodes)
    {
	$elem++ if $n->isElementNode;
    }
    
    if ($elem > 0 && defined ($self->getDocumentElement))
    {
	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "document can have only one Element");
    }
    $self->SUPER::appendChild ($node);
}

sub insertBefore
{
    my ($self, $node, $refNode) = @_;

    # Extra check: make sure sure we don't end up with more than 1 Elements.
    # Don't worry about multiple DocType nodes, because DocumentFragment
    # can't contain DocType nodes.

    my @nodes = ($node);
    @nodes = @{$node->[_C]}
	if $node->getNodeType == DOCUMENT_FRAGMENT_NODE;
    
    my $elem = 0;
    for my $n (@nodes)
    {
	$elem++ if $n->isElementNode;
    }
    
    if ($elem > 0 && defined ($self->getDocumentElement))
    {
	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "document can have only one Element");
    }
    $self->SUPER::insertBefore ($node, $refNode);
}

sub replaceChild
{
    my ($self, $node, $refNode) = @_;

    # Extra check: make sure sure we don't end up with more than 1 Elements.
    # Don't worry about multiple DocType nodes, because DocumentFragment
    # can't contain DocType nodes.

    my @nodes = ($node);
    @nodes = @{$node->[_C]}
	if $node->getNodeType == DOCUMENT_FRAGMENT_NODE;
    
    my $elem = 0;
    $elem-- if $refNode->isElementNode;

    for my $n (@nodes)
    {
	$elem++ if $n->isElementNode;
    }
    
    if ($elem > 0 && defined ($self->getDocumentElement))
    {
	croak new XML::DOM::DOMException (HIERARCHY_REQUEST_ERR,
					  "document can have only one Element");
    }
    $self->SUPER::replaceChild ($node, $refNode);
}

#------------------------------------------------------------
# Extra method implementations

sub isReadOnly
{
    0;
}

sub print
{
    my ($self, $FILE) = @_;

    my $xmlDecl = $self->getXMLDecl;
    if (defined $xmlDecl)
    {
	$xmlDecl->print ($FILE);
	$FILE->print ("\x0A");
    }

    for my $node (@{$self->[_C]})
    {
	$node->print ($FILE);
	$FILE->print ("\x0A");
    }
}

sub setDoctype
{
    my ($self, $doctype) = @_;
    my $oldDoctype = $self->[_Doctype];
    if (defined $oldDoctype)
    {
	$self->replaceChild ($doctype, $oldDoctype);
    }
    else
    {
#?? before root element, but after XmlDecl !
	$self->appendChild ($doctype);
    }
    $_[0]->[_Doctype] = $_[1];
}

sub removeDoctype
{
    my $self = shift;
    my $doctype = $self->removeChild ($self->[_Doctype]);

    undef $self->[_Doctype]; # was delete
    $doctype;
}

sub rejectChild
{
    my $t = $_[1]->getNodeType;
    $t != ELEMENT_NODE
	&& $t != PROCESSING_INSTRUCTION_NODE
	&& $t != COMMENT_NODE
	&& $t != DOCUMENT_TYPE_NODE;
}

sub expandEntity
{
    my ($self, $ent, $param) = @_;
    my $doctype = $self->getDoctype;

    (defined $doctype) ? $doctype->expandEntity ($ent, $param) : undef;
}

sub getDefaultAttrValue
{
    my ($self, $elem, $attr) = @_;
    
    my $doctype = $self->getDoctype;

    (defined $doctype) ? $doctype->getDefaultAttrValue ($elem, $attr) : undef;
}

sub getEntity
{
    my ($self, $entity) = @_;
    
    my $doctype = $self->getDoctype;

    (defined $doctype) ? $doctype->getEntity ($entity) : undef;
}

sub dispose
{
    my $self = shift;

    $self->[_XmlDecl]->dispose if defined $self->[_XmlDecl];
    undef $self->[_XmlDecl]; # was delete
    undef $self->[_Doctype]; # was delete
    $self->SUPER::dispose;
}

sub setOwnerDocument
{
    # Do nothing, you can't change the owner document!
#?? could throw exception...
}

sub getXMLDecl
{
    $_[0]->[_XmlDecl];
}

sub setXMLDecl
{
    $_[0]->[_XmlDecl] = $_[1];
}

sub createXMLDecl
{
    new XML::DOM::XMLDecl (@_);
}

sub createNotation
{
    new XML::DOM::Notation (@_);
}

sub createElementDecl
{
    new XML::DOM::ElementDecl (@_);
}

sub createAttlistDecl
{
    new XML::DOM::AttlistDecl (@_);
}

sub createEntity
{
    new XML::DOM::Entity (@_);
}

sub createChecker
{
    my $self = shift;
    my $checker = XML::Checker->new;

    $checker->Init;
    my $doctype = $self->getDoctype;
    $doctype->to_expat ($checker) if $doctype;
    $checker->Final;

    $checker;
}

sub check
{
    my ($self, $checker) = @_;
    $checker ||= XML::Checker->new;

    $self->to_expat ($checker);
}

sub to_expat
{
    my ($self, $iter) = @_;

    $iter->Init;

    for my $kid ($self->getChildNodes)
    {
	$kid->to_expat ($iter);
    }
    $iter->Final;
}

sub check_sax
{
    my ($self, $checker) = @_;
    $checker ||= XML::Checker->new;

    $self->to_sax (Handler => $checker);
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;

    $doch->start_document;

    for my $kid ($self->getChildNodes)
    {
	$kid->_to_sax ($doch, $dtdh, $enth);
    }
    $doch->end_document;
}

######################################################################
package XML::DOM::DocumentType;
######################################################################
use vars qw{ @ISA @EXPORT_OK %EXPORT_TAGS %HFIELDS };

BEGIN
{
    import XML::DOM::Node qw( :DEFAULT :Fields );
    import XML::DOM::Document qw( :Fields );
    XML::DOM::def_fields ("Entities Notations Name SysId PubId Internal", "XML::DOM::Node");
}

use XML::DOM::DOMException;
use XML::DOM::NamedNodeMap;

sub new
{
    my $class = shift;
    my $doc = shift;

    my $self = bless [], $class;

    $self->[_Doc] = $doc;
    $self->[_ReadOnly] = 1;
    $self->[_C] = new XML::DOM::NodeList;

    $self->[_Entities] =  new XML::DOM::NamedNodeMap (Doc	=> $doc,
						      Parent	=> $self,
						      ReadOnly	=> 1);
    $self->[_Notations] = new XML::DOM::NamedNodeMap (Doc	=> $doc,
						      Parent	=> $self,
						      ReadOnly	=> 1);
    $self->setParams (@_);
    $self;
}

sub getNodeType
{
    DOCUMENT_TYPE_NODE;
}

sub getNodeName
{
    $_[0]->[_Name];
}

sub getName
{
    $_[0]->[_Name];
}

sub getEntities
{
    $_[0]->[_Entities];
}

sub getNotations
{
    $_[0]->[_Notations];
}

sub setParentNode
{
    my ($self, $parent) = @_;
    $self->SUPER::setParentNode ($parent);

    $parent->[_Doctype] = $self 
	if $parent->getNodeType == DOCUMENT_NODE;
}

sub cloneNode
{
    my ($self, $deep) = @_;

    my $node = new XML::DOM::DocumentType ($self->[_Doc], $self->[_Name], 
					   $self->[_SysId], $self->[_PubId], 
					   $self->[_Internal]);

#?? does it make sense to make a shallow copy?

    # clone the NamedNodeMaps
    $node->[_Entities] = $self->[_Entities]->cloneNode ($deep);

    $node->[_Notations] = $self->[_Notations]->cloneNode ($deep);

    $node->cloneChildren ($self, $deep);

    $node;
}

#------------------------------------------------------------
# Extra method implementations

sub getSysId
{
    $_[0]->[_SysId];
}

sub getPubId
{
    $_[0]->[_PubId];
}

sub getInternal
{
    $_[0]->[_Internal];
}

sub setSysId
{
    $_[0]->[_SysId] = $_[1];
}

sub setPubId
{
    $_[0]->[_PubId] = $_[1];
}

sub setInternal
{
    $_[0]->[_Internal] = $_[1];
}

sub setName
{
    $_[0]->[_Name] = $_[1];
}

sub removeChildHoodMemories
{
    my ($self, $dontWipeReadOnly) = @_;

    my $parent = $self->[_Parent];
    if (defined $parent && $parent->getNodeType == DOCUMENT_NODE)
    {
	undef $parent->[_Doctype]; # was delete
    }
    $self->SUPER::removeChildHoodMemories;
}

sub dispose
{
    my $self = shift;

    $self->[_Entities]->dispose;
    $self->[_Notations]->dispose;
    $self->SUPER::dispose;
}

sub setOwnerDocument
{
    my ($self, $doc) = @_;
    $self->SUPER::setOwnerDocument ($doc);

    $self->[_Entities]->setOwnerDocument ($doc);
    $self->[_Notations]->setOwnerDocument ($doc);
}

sub expandEntity
{
    my ($self, $ent, $param) = @_;

    my $kid = $self->[_Entities]->getNamedItem ($ent);
    return $kid->getValue
	if (defined ($kid) && $param == $kid->isParameterEntity);

    undef;	# entity not found
}

sub getAttlistDecl
{
    my ($self, $elemName) = @_;
    for my $kid (@{$_[0]->[_C]})
    {
	return $kid if ($kid->getNodeType == ATTLIST_DECL_NODE &&
			$kid->getName eq $elemName);
    }
    undef;	# not found
}

sub getElementDecl
{
    my ($self, $elemName) = @_;
    for my $kid (@{$_[0]->[_C]})
    {
	return $kid if ($kid->getNodeType == ELEMENT_DECL_NODE &&
			$kid->getName eq $elemName);
    }
    undef;	# not found
}

sub addElementDecl
{
    my ($self, $name, $model, $hidden) = @_;
    my $node = $self->getElementDecl ($name);

#?? could warn
    unless (defined $node)
    {
	$node = $self->[_Doc]->createElementDecl ($name, $model, $hidden);
	$self->appendChild ($node);
    }
    $node;
}

sub addAttlistDecl
{
    my ($self, $name) = @_;
    my $node = $self->getAttlistDecl ($name);

    unless (defined $node)
    {
	$node = $self->[_Doc]->createAttlistDecl ($name);
	$self->appendChild ($node);
    }
    $node;
}

sub addNotation
{
    my $self = shift;
    my $node = $self->[_Doc]->createNotation (@_);
    $self->[_Notations]->setNamedItem ($node);
    $node;
}

sub addEntity
{
    my $self = shift;
    my $node = $self->[_Doc]->createEntity (@_);

    $self->[_Entities]->setNamedItem ($node);
    $node;
}

# All AttDefs for a certain Element are merged into a single ATTLIST
sub addAttDef
{
    my $self = shift;
    my $elemName = shift;

    # create the AttlistDecl if it doesn't exist yet
    my $attListDecl = $self->addAttlistDecl ($elemName);
    $attListDecl->addAttDef (@_);
}

sub getDefaultAttrValue
{
    my ($self, $elem, $attr) = @_;
    my $elemNode = $self->getAttlistDecl ($elem);
    (defined $elemNode) ? $elemNode->getDefaultAttrValue ($attr) : undef;
}

sub getEntity
{
    my ($self, $entity) = @_;
    $self->[_Entities]->getNamedItem ($entity);
}

sub setParams
{
    my ($self, $name, $sysid, $pubid, $internal) = @_;

    $self->[_Name] = $name;

#?? not sure if we need to hold on to these...
    $self->[_SysId] = $sysid if defined $sysid;
    $self->[_PubId] = $pubid if defined $pubid;
    $self->[_Internal] = $internal if defined $internal;

    $self;
}

sub rejectChild
{
    # DOM Spec says: DocumentType -- no children
    not $XML::DOM::IgnoreReadOnly;
}

sub print
{
    my ($self, $FILE) = @_;

    my $name = $self->[_Name];

    my $sysId = $self->[_SysId];
    my $pubId = $self->[_PubId];

    $FILE->print ("<!DOCTYPE $name");
    if (defined $pubId)
    {
	$FILE->print (" PUBLIC \"$pubId\" \"$sysId\"");
    }
    elsif (defined $sysId)
    {
	$FILE->print (" SYSTEM \"$sysId\"");
    }

    my @entities = @{$self->[_Entities]->getValues};
    my @notations = @{$self->[_Notations]->getValues};
    my @kids = @{$self->[_C]};

    if (@entities || @notations || @kids)
    {
	$FILE->print (" [\x0A");

	for my $kid (@entities)
	{
	    next if $kid->[_Hidden];

	    $FILE->print (" ");
	    $kid->print ($FILE);
	    $FILE->print ("\x0A");
	}

	for my $kid (@notations)
	{
	    next if $kid->[_Hidden];

	    $FILE->print (" ");
	    $kid->print ($FILE);
	    $FILE->print ("\x0A");
	}

	for my $kid (@kids)
	{
	    next if $kid->[_Hidden];

	    $FILE->print (" ");
	    $kid->print ($FILE);
	    $FILE->print ("\x0A");
	}
	$FILE->print ("]");
    }
    $FILE->print (">");
}

sub to_expat
{
    my ($self, $iter) = @_;

    $iter->Doctype ($self->getName, $self->getSysId, $self->getPubId, $self->getInternal);

    for my $ent ($self->getEntities->getValues)
    {
	next if $ent->[_Hidden];
	$ent->to_expat ($iter);
    }

    for my $nota ($self->getNotations->getValues)
    {
	next if $nota->[_Hidden];
	$nota->to_expat ($iter);
    }

    for my $kid ($self->getChildNodes)
    {
	next if $kid->[_Hidden];
	$kid->to_expat ($iter);
    }
}

sub _to_sax
{
    my ($self, $doch, $dtdh, $enth) = @_;

    $dtdh->doctype_decl ( { Name => $self->getName, 
			    SystemId => $self->getSysId, 
			    PublicId => $self->getPubId, 
			    Internal => $self->getInternal });

    for my $ent ($self->getEntities->getValues)
    {
	next if $ent->[_Hidden];
	$ent->_to_sax ($doch, $dtdh, $enth);
    }

    for my $nota ($self->getNotations->getValues)
    {
	next if $nota->[_Hidden];
	$nota->_to_sax ($doch, $dtdh, $enth);
    }

    for my $kid ($self->getChildNodes)
    {
	next if $kid->[_Hidden];
	$kid->_to_sax ($doch, $dtdh, $enth);
    }
}

######################################################################
package XML::DOM::Parser;
######################################################################
use vars qw ( @ISA );
@ISA = qw( XML::Parser );

sub new
{
    my ($class, %args) = @_;

    $args{Style} = 'XML::Parser::Dom';
    $class->SUPER::new (%args);
}

# This method needed to be overriden so we can restore some global 
# variables when an exception is thrown
sub parse
{
    my $self = shift;

    local $XML::Parser::Dom::_DP_doc;
    local $XML::Parser::Dom::_DP_elem;
    local $XML::Parser::Dom::_DP_doctype;
    local $XML::Parser::Dom::_DP_in_prolog;
    local $XML::Parser::Dom::_DP_end_doc;
    local $XML::Parser::Dom::_DP_saw_doctype;
    local $XML::Parser::Dom::_DP_in_CDATA;
    local $XML::Parser::Dom::_DP_keep_CDATA;
    local $XML::Parser::Dom::_DP_last_text;


    # Temporarily disable checks that Expat already does (for performance)
    local $XML::DOM::SafeMode = 0;
    # Temporarily disable ReadOnly checks
    local $XML::DOM::IgnoreReadOnly = 1;

    my $ret;
    eval {
	$ret = $self->SUPER::parse (@_);
    };
    my $err = $@;

    if ($err)
    {
	my $doc = $XML::Parser::Dom::_DP_doc;
	if ($doc)
	{
	    $doc->dispose;
	}
	die $err;
    }

    $ret;
}

my $LWP_USER_AGENT;
sub set_LWP_UserAgent
{
    $LWP_USER_AGENT = shift;
}

sub parsefile
{
    my $self = shift;
    my $url = shift;

    # Any other URL schemes?
    if ($url =~ /^(https?|ftp|wais|gopher|file):/)
    {
	# Read the file from the web with LWP.
	#
	# Note that we read in the entire file, which may not be ideal
	# for large files. LWP::UserAgent also provides a callback style
	# request, which we could convert to a stream with a fork()...

	my $result;
	eval
	{
	    use LWP::UserAgent;

	    my $ua = $self->{LWP_UserAgent};
	    unless (defined $ua)
	    {
		unless (defined $LWP_USER_AGENT)
		{
		    $LWP_USER_AGENT = LWP::UserAgent->new;

		    # Load proxy settings from environment variables, i.e.:
		    # http_proxy, ftp_proxy, no_proxy etc. (see LWP::UserAgent(3))
		    # You need these to go thru firewalls.
		    $LWP_USER_AGENT->env_proxy;
		}
		$ua = $LWP_USER_AGENT;
	    }
	    my $req = new HTTP::Request 'GET', $url;
	    my $response = $ua->request ($req);

	    # Parse the result of the HTTP request
	    $result = $self->parse ($response->content, @_);
	};
	if ($@)
	{
	    die "Couldn't parsefile [$url] with LWP: $@";
	}
	return $result;
    }
    else
    {
	return $self->SUPER::parsefile ($url, @_);
    }
}

######################################################################
package XML::Parser::Dom;
######################################################################

BEGIN
{
    import XML::DOM::Node qw( :Fields );
    import XML::DOM::CharacterData qw( :Fields );
}

use vars qw( $_DP_doc
	     $_DP_elem
	     $_DP_doctype
	     $_DP_in_prolog
	     $_DP_end_doc
	     $_DP_saw_doctype
	     $_DP_in_CDATA
	     $_DP_keep_CDATA
	     $_DP_last_text
	     $_DP_level
	     $_DP_expand_pent
	   );

# This adds a new Style to the XML::Parser class.
# From now on you can say: $parser = new XML::Parser ('Style' => 'Dom' );
# but that is *NOT* how a regular user should use it!
$XML::Parser::Built_In_Styles{Dom} = 1;

sub Init
{
    $_DP_elem = $_DP_doc = new XML::DOM::Document();
    $_DP_doctype = new XML::DOM::DocumentType ($_DP_doc);
    $_DP_doc->setDoctype ($_DP_doctype);
    $_DP_keep_CDATA = $_[0]->{KeepCDATA};

    # Prepare for document prolog
    $_DP_in_prolog = 1;

    # We haven't passed the root element yet
    $_DP_end_doc = 0;

    # Expand parameter entities in the DTD by default

    $_DP_expand_pent = defined $_[0]->{ExpandParamEnt} ? 
					$_[0]->{ExpandParamEnt} : 1;
    if ($_DP_expand_pent)
    {
	$_[0]->{DOM_Entity} = {};
    }

    $_DP_level = 0;

    undef $_DP_last_text;
}

sub Final
{
    unless ($_DP_saw_doctype)
    {
	my $doctype = $_DP_doc->removeDoctype;
	$doctype->dispose;
    }
    $_DP_doc;
}

sub Char
{
    my $str = $_[1];

    if ($_DP_in_CDATA && $_DP_keep_CDATA)
    {
	undef $_DP_last_text;
	# Merge text with previous node if possible
	$_DP_elem->addCDATA ($str);
    }
    else
    {
	# Merge text with previous node if possible
	# Used to be:	$expat->{DOM_Element}->addText ($str);
	if ($_DP_last_text)
	{
	    $_DP_last_text->[_Data] .= $str;
	}
	else
	{
	    $_DP_last_text = $_DP_doc->createTextNode ($str);
	    $_DP_last_text->[_Parent] = $_DP_elem;
	    push @{$_DP_elem->[_C]}, $_DP_last_text;
	}
    }
}

sub Start
{
    my ($expat, $elem, @attr) = @_;
    my $parent = $_DP_elem;
    my $doc = $_DP_doc;
    
    if ($parent == $doc)
    {
	# End of document prolog, i.e. start of first Element
	$_DP_in_prolog = 0;
    }
    
    undef $_DP_last_text;
    my $node = $doc->createElement ($elem);
    $_DP_elem = $node;
    $parent->appendChild ($node);
    
    my $n = @attr;
    return unless $n;

    # Add attributes
    my $first_default = $expat->specified_attr;
    my $i = 0;
    while ($i < $n)
    {
	my $specified = $i < $first_default;
	my $name = $attr[$i++];
	undef $_DP_last_text;
	my $attr = $doc->createAttribute ($name, $attr[$i++], $specified);
	$node->setAttributeNode ($attr);
    }
}

sub End
{
    $_DP_elem = $_DP_elem->[_Parent];
    undef $_DP_last_text;

    # Check for end of root element
    $_DP_end_doc = 1 if ($_DP_elem == $_DP_doc);
}

# Called at end of file, i.e. whitespace following last closing tag
# Also for Entity references
# May also be called at other times...
sub Default
{
    my ($expat, $str) = @_;

#    shift; deb ("Default", @_);

    if ($_DP_in_prolog)	# still processing Document prolog...
    {
#?? could try to store this text later
#?? I've only seen whitespace here so far
    }
    elsif (!$_DP_end_doc)	# ignore whitespace at end of Document
    {
#	if ($expat->{NoExpand})
#	{
	    # Got a TextDecl (<?xml ...?>) from an external entity here once

	    # create non-parameter entity reference, correct?
            return unless $str =~ s!^&!!;
            return unless $str =~ s!;$!!;
	    $_DP_elem->appendChild (
		   $_DP_doc->createEntityReference ($str,0,$expat->{NoExpand}));
	    undef $_DP_last_text;
#	}
#	else
#	{
#	    $expat->{DOM_Element}->addText ($str);
#	}
    }
}

# XML::Parser 2.19 added support for CdataStart and CdataEnd handlers
# If they are not defined, the Default handler is called instead
# with the text "<![CDATA[" and "]]"
sub CdataStart
{
    $_DP_in_CDATA = 1;
}

sub CdataEnd
{
    $_DP_in_CDATA = 0;
}

my $START_MARKER = "__DOM__START__ENTITY__";
my $END_MARKER = "__DOM__END__ENTITY__";

sub Comment
{
    undef $_DP_last_text;

    # These comments were inserted by ExternEnt handler
    if ($_[1] =~ /(?:($START_MARKER)|($END_MARKER))/)
    {
	if ($1)	 # START
	{
	    $_DP_level++;
	}
	else
	{
	    $_DP_level--;
	}
    }
    else
    {
	my $comment = $_DP_doc->createComment ($_[1]);
	$_DP_elem->appendChild ($comment);
    }
}

sub deb
{
#    return;

    my $name = shift;
    print "$name (" . join(",", map {defined($_)?$_ : "(undef)"} @_) . ")\n";
}

sub Doctype
{
    my $expat = shift;
#    deb ("Doctype", @_);

    $_DP_doctype->setParams (@_);
    $_DP_saw_doctype = 1;
}

sub Attlist
{
    my $expat = shift;
#    deb ("Attlist", @_);

    $_[5] = "Hidden" unless $_DP_expand_pent || $_DP_level == 0;
    $_DP_doctype->addAttDef (@_);
}

sub XMLDecl
{
    my $expat = shift;
#    deb ("XMLDecl", @_);

    undef $_DP_last_text;
    $_DP_doc->setXMLDecl (new XML::DOM::XMLDecl ($_DP_doc, @_));
}

sub Entity
{
    my $expat = shift;
#    deb ("Entity", @_);
    
    # check to see if Parameter Entity
    if ($_[5])
    {

	if (defined $_[2])	# was sysid specified?
	{
	    # Store the Entity mapping for use in ExternEnt
	    if (exists $expat->{DOM_Entity}->{$_[2]})
	    {
		# If this ever happens, the name of entity may be the wrong one
		# when writing out the Document.
		XML::DOM::warning ("Entity $_[2] is known as %$_[0] and %" .
				   $expat->{DOM_Entity}->{$_[2]});
	    }
	    else
	    {
		$expat->{DOM_Entity}->{$_[2]} = $_[0];
	    }
	    #?? remove this block when XML::Parser has better support
	}
    }

    # no value on things with sysId
    if (defined $_[2] && defined $_[1])
    {
        # print STDERR "XML::DOM Warning $_[0] had both value($_[1]) And SYSId ($_[2]), removing value.\n";
        $_[1] = undef;
    }

    undef $_DP_last_text;

    $_[6] = "Hidden" unless $_DP_expand_pent || $_DP_level == 0;
    $_DP_doctype->addEntity (@_);
}

#
# Unparsed is called when it encounters e.g:
#
#   <!ENTITY logo SYSTEM "http://server/logo.gif" NDATA gif>
#
sub Unparsed
{
    Entity (@_);	# same as regular ENTITY, as far as DOM is concerned
}

sub Element
{
    shift;
#    deb ("Element", @_);

    # put in to convert XML::Parser::ContentModel object to string
    # ($_[1] used to be a string in XML::Parser 2.27 and
    # dom_attr.t fails if we don't stringify here)
    $_[1] = "$_[1]";

    undef $_DP_last_text;
    push @_, "Hidden" unless $_DP_expand_pent || $_DP_level == 0;
    $_DP_doctype->addElementDecl (@_);
}

sub Notation
{
    shift;
#    deb ("Notation", @_);

    undef $_DP_last_text;
    $_[4] = "Hidden" unless $_DP_expand_pent || $_DP_level == 0;
    $_DP_doctype->addNotation (@_);
}

sub Proc
{
    shift;
#    deb ("Proc", @_);

    undef $_DP_last_text;
    push @_, "Hidden" unless $_DP_expand_pent || $_DP_level == 0;
    $_DP_elem->appendChild ($_DP_doc->createProcessingInstruction (@_));
}

#
# ExternEnt is called when an external entity, such as:
#
#	<!ENTITY externalEntity PUBLIC "-//Enno//TEXT Enno's description//EN" 
#	                        "http://server/descr.txt">
#
# is referenced in the document, e.g. with: &externalEntity;
# If ExternEnt is not specified, the entity reference is passed to the Default
# handler as e.g. "&externalEntity;", where an EntityReference object is added.
#
# Also for %externalEntity; references in the DTD itself.
#
# It can also be called when XML::Parser parses the DOCTYPE header
# (just before calling the DocType handler), when it contains a
# reference like "docbook.dtd" below:
#
#    <!DOCTYPE book PUBLIC "-//Norman Walsh//DTD DocBk XML V3.1.3//EN" 
#	"docbook.dtd" [
#     ... rest of DTD ...
#
sub ExternEnt
{
    my ($expat, $base, $sysid, $pubid) = @_;
#    deb ("ExternEnt", @_);

    # ?? (tjmather) i think there is a problem here
    # with XML::Parser > 2.27 since file_ext_ent_handler
    # now returns a IO::File object instead of a content string

    # Invoke XML::Parser's default ExternEnt handler
    my $content;
    if ($XML::Parser::have_LWP)
    {
	$content = XML::Parser::lwp_ext_ent_handler (@_);
    }
    else
    {
	$content = XML::Parser::file_ext_ent_handler (@_);
    }

    if ($_DP_expand_pent)
    {
	return $content;
    }
    else
    {
	my $entname = $expat->{DOM_Entity}->{$sysid};
	if (defined $entname)
	{
	    $_DP_doctype->appendChild ($_DP_doc->createEntityReference ($entname, 1, $expat->{NoExpand}));
            # Wrap the contents in special comments, so we know when we reach the
	    # end of parsing the entity. This way we can omit the contents from
	    # the DTD, when ExpandParamEnt is set to 0.
     
	    return "<!-- $START_MARKER sysid=[$sysid] -->" .
		$content . "<!-- $END_MARKER sysid=[$sysid] -->";
	}
	else
	{
	    # We either read the entity ref'd by the system id in the 
	    # <!DOCTYPE> header, or the entity was undefined.
	    # In either case, don't bother with maintaining the entity
	    # reference, just expand the contents.
	    return "<!-- $START_MARKER sysid=[DTD] -->" .
		$content . "<!-- $END_MARKER sysid=[DTD] -->";
	}
    }
}

1; # module return code

__END__

=head1 NAME

XML::DOM - A perl module for building DOM Level 1 compliant document structures

=head1 SYNOPSIS

 use XML::DOM;

 my $parser = new XML::DOM::Parser;
 my $doc = $parser->parsefile ("file.xml");

 # print all HREF attributes of all CODEBASE elements
 my $nodes = $doc->getElementsByTagName ("CODEBASE");
 my $n = $nodes->getLength;

 for (my $i = 0; $i < $n; $i++)
 {
     my $node = $nodes->item ($i);
     my $href = $node->getAttributeNode ("HREF");
     print $href->getValue . "\n";
 }

 # Print doc file
 $doc->printToFile ("out.xml");

 # Print to string
 print $doc->toString;

 # Avoid memory leaks - cleanup circular references for garbage collection
 $doc->dispose;

=head1 DESCRIPTION

This module extends the XML::Parser module by Clark Cooper. 
The XML::Parser module is built on top of XML::Parser::Expat, 
which is a lower level interface to James Clark's expat library.

XML::DOM::Parser is derived from XML::Parser. It parses XML strings or files
and builds a data structure that conforms to the API of the Document Object 
Model as described at http://www.w3.org/TR/REC-DOM-Level-1.
See the XML::Parser manpage for other available features of the 
XML::DOM::Parser class. 
Note that the 'Style' property should not be used (it is set internally.)

The XML::Parser I<NoExpand> option is more or less supported, in that it will
generate EntityReference objects whenever an entity reference is encountered
in character data. I'm not sure how useful this is. Any comments are welcome.

As described in the synopsis, when you create an XML::DOM::Parser object, 
the parse and parsefile methods create an I<XML::DOM::Document> object
from the specified input. This Document object can then be examined, modified and
written back out to a file or converted to a string.

When using XML::DOM with XML::Parser version 2.19 and up, setting the 
XML::DOM::Parser option I<KeepCDATA> to 1 will store CDATASections in
CDATASection nodes, instead of converting them to Text nodes.
Subsequent CDATASection nodes will be merged into one. Let me know if this
is a problem.

When using XML::Parser 2.27 and above, you can suppress expansion of
parameter entity references (e.g. %pent;) in the DTD, by setting I<ParseParamEnt>
to 1 and I<ExpandParamEnt> to 0. See L<Hidden Nodes|/_Hidden_Nodes_> for details.

A Document has a tree structure consisting of I<Node> objects. A Node may contain
other nodes, depending on its type.
A Document may have Element, Text, Comment, and CDATASection nodes. 
Element nodes may have Attr, Element, Text, Comment, and CDATASection nodes. 
The other nodes may not have any child nodes. 

This module adds several node types that are not part of the DOM spec (yet.)
These are: ElementDecl (for <!ELEMENT ...> declarations), AttlistDecl (for
<!ATTLIST ...> declarations), XMLDecl (for <?xml ...?> declarations) and AttDef
(for attribute definitions in an AttlistDecl.)

=head1 XML::DOM Classes

The XML::DOM module stores XML documents in a tree structure with a root node
of type XML::DOM::Document. Different nodes in tree represent different
parts of the XML file. The DOM Level 1 Specification defines the following
node types:

=over 4

=item * L<XML::DOM::Node> - Super class of all node types

=item * L<XML::DOM::Document> - The root of the XML document

=item * L<XML::DOM::DocumentType> - Describes the document structure: <!DOCTYPE root [ ... ]>

=item * L<XML::DOM::Element> - An XML element: <elem attr="val"> ... </elem>

=item * L<XML::DOM::Attr> - An XML element attribute: name="value"

=item * L<XML::DOM::CharacterData> - Super class of Text, Comment and CDATASection

=item * L<XML::DOM::Text> - Text in an XML element

=item * L<XML::DOM::CDATASection> - Escaped block of text: <![CDATA[ text ]]>

=item * L<XML::DOM::Comment> - An XML comment: <!-- comment -->

=item * L<XML::DOM::EntityReference> - Refers to an ENTITY: &ent; or %ent;

=item * L<XML::DOM::Entity> - An ENTITY definition: <!ENTITY ...>

=item * L<XML::DOM::ProcessingInstruction> - <?PI target>

=item * L<XML::DOM::DocumentFragment> - Lightweight node for cut & paste

=item * L<XML::DOM::Notation> - An NOTATION definition: <!NOTATION ...>

=back

In addition, the XML::DOM module contains the following nodes that are not part 
of the DOM Level 1 Specification:

=over 4

=item * L<XML::DOM::ElementDecl> - Defines an element: <!ELEMENT ...>

=item * L<XML::DOM::AttlistDecl> - Defines one or more attributes in an <!ATTLIST ...>

=item * L<XML::DOM::AttDef> - Defines one attribute in an <!ATTLIST ...>

=item * L<XML::DOM::XMLDecl> - An XML declaration: <?xml version="1.0" ...>

=back

Other classes that are part of the DOM Level 1 Spec:

=over 4

=item * L<XML::DOM::Implementation> - Provides information about this implementation. Currently it doesn't do much.

=item * L<XML::DOM::NodeList> - Used internally to store a node's child nodes. Also returned by getElementsByTagName.

=item * L<XML::DOM::NamedNodeMap> - Used internally to store an element's attributes.

=back

Other classes that are not part of the DOM Level 1 Spec:

=over 4

=item * L<XML::DOM::Parser> - An non-validating XML parser that creates XML::DOM::Documents

=item * L<XML::DOM::ValParser> - A validating XML parser that creates XML::DOM::Documents. It uses L<XML::Checker> to check against the DocumentType (DTD)

=item * L<XML::Handler::BuildDOM> - A PerlSAX handler that creates XML::DOM::Documents.

=back

=head1 XML::DOM package

=over 4

=item Constant definitions

The following predefined constants indicate which type of node it is.

=back

 UNKNOWN_NODE (0)                The node type is unknown (not part of DOM)

 ELEMENT_NODE (1)                The node is an Element.
 ATTRIBUTE_NODE (2)              The node is an Attr.
 TEXT_NODE (3)                   The node is a Text node.
 CDATA_SECTION_NODE (4)          The node is a CDATASection.
 ENTITY_REFERENCE_NODE (5)       The node is an EntityReference.
 ENTITY_NODE (6)                 The node is an Entity.
 PROCESSING_INSTRUCTION_NODE (7) The node is a ProcessingInstruction.
 COMMENT_NODE (8)                The node is a Comment.
 DOCUMENT_NODE (9)               The node is a Document.
 DOCUMENT_TYPE_NODE (10)         The node is a DocumentType.
 DOCUMENT_FRAGMENT_NODE (11)     The node is a DocumentFragment.
 NOTATION_NODE (12)              The node is a Notation.

 ELEMENT_DECL_NODE (13)		 The node is an ElementDecl (not part of DOM)
 ATT_DEF_NODE (14)		 The node is an AttDef (not part of DOM)
 XML_DECL_NODE (15)		 The node is an XMLDecl (not part of DOM)
 ATTLIST_DECL_NODE (16)		 The node is an AttlistDecl (not part of DOM)

 Usage:

   if ($node->getNodeType == ELEMENT_NODE)
   {
       print "It's an Element";
   }

B<Not In DOM Spec>: The DOM Spec does not mention UNKNOWN_NODE and, 
quite frankly, you should never encounter it. The last 4 node types were added
to support the 4 added node classes.

=head2 Global Variables

=over 4

=item $VERSION

The variable $XML::DOM::VERSION contains the version number of this 
implementation, e.g. "1.43".

=back

=head2 METHODS

These methods are not part of the DOM Level 1 Specification.

=over 4

=item getIgnoreReadOnly and ignoreReadOnly (readOnly)

The DOM Level 1 Spec does not allow you to edit certain sections of the document,
e.g. the DocumentType, so by default this implementation throws DOMExceptions
(i.e. NO_MODIFICATION_ALLOWED_ERR) when you try to edit a readonly node. 
These readonly checks can be disabled by (temporarily) setting the global 
IgnoreReadOnly flag.

The ignoreReadOnly method sets the global IgnoreReadOnly flag and returns its
previous value. The getIgnoreReadOnly method simply returns its current value.

 my $oldIgnore = XML::DOM::ignoreReadOnly (1);
 eval {
 ... do whatever you want, catching any other exceptions ...
 };
 XML::DOM::ignoreReadOnly ($oldIgnore);     # restore previous value

Another way to do it, using a local variable:

 { # start new scope
    local $XML::DOM::IgnoreReadOnly = 1;
    ... do whatever you want, don't worry about exceptions ...
 } # end of scope ($IgnoreReadOnly is set back to its previous value)
    

=item isValidName (name)

Whether the specified name is a valid "Name" as specified in the XML spec.
Characters with Unicode values > 127 are now also supported.

=item getAllowReservedNames and allowReservedNames (boolean)

The first method returns whether reserved names are allowed. 
The second takes a boolean argument and sets whether reserved names are allowed.
The initial value is 1 (i.e. allow reserved names.)

The XML spec states that "Names" starting with (X|x)(M|m)(L|l)
are reserved for future use. (Amusingly enough, the XML version of the XML spec
(REC-xml-19980210.xml) breaks that very rule by defining an ENTITY with the name 
'xmlpio'.)
A "Name" in this context means the Name token as found in the BNF rules in the
XML spec.

XML::DOM only checks for errors when you modify the DOM tree, not when the
DOM tree is built by the XML::DOM::Parser.

=item setTagCompression (funcref)

There are 3 possible styles for printing empty Element tags:

=over 4

=item Style 0

 <empty/> or <empty attr="val"/>

XML::DOM uses this style by default for all Elements.

=item Style 1

  <empty></empty> or <empty attr="val"></empty>

=item Style 2

  <empty /> or <empty attr="val" />

This style is sometimes desired when using XHTML. 
(Note the extra space before the slash "/")
See L<http://www.w3.org/TR/xhtml1> Appendix C for more details.

=back

By default XML::DOM compresses all empty Element tags (style 0.)
You can control which style is used for a particular Element by calling
XML::DOM::setTagCompression with a reference to a function that takes
2 arguments. The first is the tag name of the Element, the second is the
XML::DOM::Element that is being printed. 
The function should return 0, 1 or 2 to indicate which style should be used to
print the empty tag. E.g.

 XML::DOM::setTagCompression (\&my_tag_compression);

 sub my_tag_compression
 {
    my ($tag, $elem) = @_;

    # Print empty br, hr and img tags like this: <br />
    return 2 if $tag =~ /^(br|hr|img)$/;

    # Print other empty tags like this: <empty></empty>
    return 1;
 }

=back

=head1 IMPLEMENTATION DETAILS

=over 4

=item * Perl Mappings

The value undef was used when the DOM Spec said null.

The DOM Spec says: Applications must encode DOMString using UTF-16 (defined in 
Appendix C.3 of [UNICODE] and Amendment 1 of [ISO-10646]).
In this implementation we use plain old Perl strings encoded in UTF-8 instead of
UTF-16.

=item * Text and CDATASection nodes

The Expat parser expands EntityReferences and CDataSection sections to 
raw strings and does not indicate where it was found. 
This implementation does therefore convert both to Text nodes at parse time.
CDATASection and EntityReference nodes that are added to an existing Document 
(by the user) will be preserved.

Also, subsequent Text nodes are always merged at parse time. Text nodes that are 
added later can be merged with the normalize method. Consider using the addText
method when adding Text nodes.

=item * Printing and toString

When printing (and converting an XML Document to a string) the strings have to 
encoded differently depending on where they occur. E.g. in a CDATASection all 
substrings are allowed except for "]]>". In regular text, certain characters are
not allowed, e.g. ">" has to be converted to "&gt;". 
These routines should be verified by someone who knows the details.

=item * Quotes

Certain sections in XML are quoted, like attribute values in an Element.
XML::Parser strips these quotes and the print methods in this implementation 
always uses double quotes, so when parsing and printing a document, single quotes
may be converted to double quotes. The default value of an attribute definition
(AttDef) in an AttlistDecl, however, will maintain its quotes.

=item * AttlistDecl

Attribute declarations for a certain Element are always merged into a single
AttlistDecl object.

=item * Comments

Comments in the DOCTYPE section are not kept in the right place. They will become
child nodes of the Document.

=item * Hidden Nodes

Previous versions of XML::DOM would expand parameter entity references
(like B<%pent;>), so when printing the DTD, it would print the contents
of the external entity, instead of the parameter entity reference.
With this release (1.27), you can prevent this by setting the XML::DOM::Parser
options ParseParamEnt => 1 and ExpandParamEnt => 0.

When it is parsing the contents of the external entities, it *DOES* still add
the nodes to the DocumentType, but it marks these nodes by setting
the 'Hidden' property. In addition, it adds an EntityReference node to the
DocumentType node.

When printing the DocumentType node (or when using to_expat() or to_sax()), 
the 'Hidden' nodes are suppressed, so you will see the parameter entity
reference instead of the contents of the external entities. See test case
t/dom_extent.t for an example.

The reason for adding the 'Hidden' nodes to the DocumentType node, is that
the nodes may contain <!ENTITY> definitions that are referenced further
in the document. (Simply not adding the nodes to the DocumentType could
cause such entity references to be expanded incorrectly.)

Note that you need XML::Parser 2.27 or higher for this to work correctly.

=back

=head1 SEE ALSO

L<XML::DOM::XPath>

The Japanese version of this document by Takanori Kawai (Hippo2000)
at L<http://member.nifty.ne.jp/hippo2000/perltips/xml/dom.htm>

The DOM Level 1 specification at L<http://www.w3.org/TR/REC-DOM-Level-1>

The XML spec (Extensible Markup Language 1.0) at L<http://www.w3.org/TR/REC-xml>

The L<XML::Parser> and L<XML::Parser::Expat> manual pages.

L<XML::LibXML> also provides a DOM Parser, and is significantly faster
than XML::DOM, and is under active development.  It requires that you 
download the Gnome libxml library.

L<XML::GDOME> will provide the DOM Level 2 Core API, and should be
as fast as XML::LibXML, but more robust, since it uses the memory
management functions of libgdome.  For more details see
L<http://tjmather.com/xml-gdome/>

=head1 CAVEATS

The method getElementsByTagName() does not return a "live" NodeList.
Whether this is an actual caveat is debatable, but a few people on the 
www-dom mailing list seemed to think so. I haven't decided yet. It's a pain
to implement, it slows things down and the benefits seem marginal.
Let me know what you think. 

=head1 AUTHOR

Enno Derksen is the original author.

Send patches to T.J. Mather at <F<tjmather@maxmind.com>>.

Paid support is available from directly from the maintainers of this package.
Please see L<http://www.maxmind.com/app/opensourceservices> for more details.

Thanks to Clark Cooper for his help with the initial version.

=cut
