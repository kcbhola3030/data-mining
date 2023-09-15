import React from 'react';

const DecisionTreeTable = ({ text }) => {
      // Replace '\n' with '<br />' to create line breaks
  const textWithLineBreaks = text.replace(/\n/g, '<br />');

  return (
    <div>
      <div dangerouslySetInnerHTML={{ __html: textWithLineBreaks }} />
    </div>
  );

}
export default DecisionTreeTable;
