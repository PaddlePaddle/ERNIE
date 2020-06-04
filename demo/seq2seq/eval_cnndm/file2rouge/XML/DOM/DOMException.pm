######################################################################
package XML::DOM::DOMException;
######################################################################

use Exporter;

use overload '""' => \&stringify;
use vars qw ( @ISA @EXPORT @ErrorNames );

BEGIN
{
  @ISA = qw( Exporter );
  @EXPORT = qw( INDEX_SIZE_ERR
		DOMSTRING_SIZE_ERR
		HIERARCHY_REQUEST_ERR
		WRONG_DOCUMENT_ERR
		INVALID_CHARACTER_ERR
		NO_DATA_ALLOWED_ERR
		NO_MODIFICATION_ALLOWED_ERR
		NOT_FOUND_ERR
		NOT_SUPPORTED_ERR
		INUSE_ATTRIBUTE_ERR
	      );
}

sub UNKNOWN_ERR			() {0;}	# not in the DOM Spec!
sub INDEX_SIZE_ERR		() {1;}
sub DOMSTRING_SIZE_ERR		() {2;}
sub HIERARCHY_REQUEST_ERR	() {3;}
sub WRONG_DOCUMENT_ERR		() {4;}
sub INVALID_CHARACTER_ERR	() {5;}
sub NO_DATA_ALLOWED_ERR		() {6;}
sub NO_MODIFICATION_ALLOWED_ERR	() {7;}
sub NOT_FOUND_ERR		() {8;}
sub NOT_SUPPORTED_ERR		() {9;}
sub INUSE_ATTRIBUTE_ERR		() {10;}

@ErrorNames = (
	       "UNKNOWN_ERR",
	       "INDEX_SIZE_ERR",
	       "DOMSTRING_SIZE_ERR",
	       "HIERARCHY_REQUEST_ERR",
	       "WRONG_DOCUMENT_ERR",
	       "INVALID_CHARACTER_ERR",
	       "NO_DATA_ALLOWED_ERR",
	       "NO_MODIFICATION_ALLOWED_ERR",
	       "NOT_FOUND_ERR",
	       "NOT_SUPPORTED_ERR",
	       "INUSE_ATTRIBUTE_ERR"
	      );
sub new
{
    my ($type, $code, $msg) = @_;
    my $self = bless {Code => $code}, $type;

    $self->{Message} = $msg if defined $msg;

#    print "=> Exception: " . $self->stringify . "\n"; 
    $self;
}

sub getCode
{
    $_[0]->{Code};
}

#------------------------------------------------------------
# Extra method implementations

sub getName
{
    $ErrorNames[$_[0]->{Code}];
}

sub getMessage
{
    $_[0]->{Message};
}

sub stringify
{
    my $self = shift;

    "XML::DOM::DOMException(Code=" . $self->getCode . ", Name=" .
	$self->getName . ", Message=" . $self->getMessage . ")";
}

1; # package return code
