#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "binary_converter.h"

// Convert a string to binary representation
char* to_binary(const char* input) {
    if (!input) return NULL;

    size_t length = strlen(input);
    char* binary = (char*)malloc(length * 8 + 1); // 8 bits per character + null terminator
    if (!binary) return NULL;

    binary[0] = '\0'; // Initialize as empty string
    for (size_t i = 0; i < length; i++) {
        char temp[9]; // 8 bits + null terminator
        snprintf(temp, sizeof(temp), "%08d", input[i] & 0xFF);
        strcat(binary, temp);
    }

    return binary;
}

// Convert binary representation back to a string
char* from_binary(const char* binary) {
    if (!binary) return NULL;

    size_t length = strlen(binary);
    if (length % 8 != 0) return NULL; // Binary string length must be a multiple of 8

    size_t text_length = length / 8; // Each character is represented by 8 bits
    char* text = (char*)malloc(text_length + 1); // +1 for null terminator
    if (!text) return NULL;

    for (size_t i = 0; i < text_length; i++) {
        char temp[9]; // 8 bits + null terminator
        strncpy(temp, &binary[i * 8], 8);
        temp[8] = '\0'; // Null-terminate the binary substring
        text[i] = (char)strtol(temp, NULL, 2);
    }
    text[text_length] = '\0'; // Null-terminate the text string

    return text;
}
