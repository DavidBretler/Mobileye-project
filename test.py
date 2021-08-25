# Python function for determining if a
# string has
# more vowels than consonants.
def if_more_vowels(string):
    def count_vowels(string):
        l = len(string)
        if l == 1:
            if string[0] in [1,5,9]:
                return True
            return False
        vowels_a = count_vowels(string[0:l//2])
        vowels_b = count_vowels(string[l//2:l])
        return vowels_a or vowels_b
    return count_vowels(string)

    if vowels > len(string) - vowels:
        return True
    return False

print(if_more_vowels([9,8,2]))

str2='auertlpuaii'
mone1=0
mone2=0

def more_vows(str1,str2):
    global mone1
    global mone2
    if len(str2)==0:
        return mone1>mone2
    if str1 in ['i','e','o','u','a']:
        mone1+=1
    else:
        mone2+=1
    return more_vows(str2[0], str2[1:])

#print(more_vows(str2[0],str2[1:]))


