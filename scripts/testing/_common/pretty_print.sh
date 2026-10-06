#!/usr/bin/env bash
# _common/pretty_print.sh
# Source this file for colored print helpers.
#
# Usage:
#   source "$SCRIPT_DIR/../_common/pretty_print.sh"
#   print_header "Starting job"
#   print_info "Running experiment: foo"
#   print_success "Finished: foo"
#   print_warn "No metrics dir found"
#   print_error "Unknown baseline"

# Colors (disabled automatically if not a terminal)
if [[ -t 1 ]]; then
  _BOLD="\033[1m"
  _RESET="\033[0m"
  _CYAN="\033[1;36m"
  _GREEN="\033[1;32m"
  _YELLOW="\033[1;33m"
  _RED="\033[1;31m"
  _BLUE="\033[1;34m"
  _GRAY="\033[0;37m"
else
  _BOLD="" _RESET="" _CYAN="" _GREEN="" _YELLOW="" _RED="" _BLUE="" _GRAY=""
fi

print_header() { echo -e "\n${_CYAN}${_BOLD}══════════════════════════════════════════${_RESET}"; \
                 echo -e "${_CYAN}${_BOLD}  $*${_RESET}"; \
                 echo -e "${_CYAN}${_BOLD}══════════════════════════════════════════${_RESET}"; }
print_info()    { echo -e "${_BLUE}[$(date '+%H:%M:%S')]${_RESET} $*"; }
print_success() { echo -e "${_GREEN}[$(date '+%H:%M:%S')] ✔  $*${_RESET}"; }
print_warn()    { echo -e "${_YELLOW}[$(date '+%H:%M:%S')] ⚠  $*${_RESET}"; }
print_error()   { echo -e "${_RED}[$(date '+%H:%M:%S')] ✘  $*${_RESET}"; }
print_kv()      { echo -e "  ${_GRAY}$1:${_RESET} ${_BOLD}$2${_RESET}"; }

